from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from models.energy import DB_PATH, initialize_database, populate_sample_data, query_rows


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "data" / "documents"
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"


def _split_text_into_chunks(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """Split source documents into overlapping chunks before vector indexing."""

    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        split_at = cleaned.rfind("\n", start, end)
        if split_at <= start:
            split_at = cleaned.rfind(". ", start, end)
        if split_at <= start:
            split_at = end
        chunk = cleaned[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = max(split_at - chunk_overlap, split_at)
        if start >= len(cleaned) or split_at == len(cleaned):
            break
    return chunks or [cleaned]


def _stable_float(text: str, low: float, high: float) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return round(low + (high - low) * value, 3)


def _simple_embedding(text: str) -> list[float]:
    """Deterministic local embedding for ChromaDB offline use."""

    dims = 64
    vector = [0.0] * dims
    words = [word.strip(".,:;!?()[]").lower() for word in text.split()]
    for word in words:
        if not word:
            continue
        idx = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % dims
        vector[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _ensure_data_ready() -> None:
    populate_sample_data(DB_PATH)


@tool
def get_weather_forecast(date: str = "tomorrow") -> dict[str, Any]:
    """Return a mock weather forecast for energy optimization.

    Args:
        date: Natural language date or ISO date, such as "tomorrow" or "2026-05-15".
    """

    conditions = ["sunny", "partly_cloudy", "cloudy", "sunny", "rainy", "sunny", "partly_cloudy"]
    base = datetime(2026, 5, 15)
    forecast = []
    for i in range(5):
        condition = conditions[(i + len(date)) % len(conditions)]
        solar_factor = {"sunny": 1.0, "partly_cloudy": 0.72, "cloudy": 0.45, "rainy": 0.25}[condition]
        forecast.append(
            {
                "date": (base + timedelta(days=i)).date().isoformat(),
                "condition": condition,
                "high_temp_f": int(_stable_float(date + str(i), 72, 94)),
                "solar_factor": solar_factor,
                "best_solar_window": "10:00-15:00" if solar_factor >= 0.7 else "11:00-13:00",
            }
        )
    return {"requested_date": date, "forecast": forecast}


@tool
def get_electricity_prices(date: str = "tomorrow") -> dict[str, Any]:
    """Return time-of-day electricity prices for a requested date.

    Args:
        date: Natural language date or ISO date.
    """

    hourly = []
    for hour in range(24):
        if hour < 7:
            price = 0.13
            period = "off_peak"
        elif hour < 16:
            price = 0.21
            period = "mid_peak"
        elif hour < 21:
            price = 0.39
            period = "peak"
        else:
            price = 0.17
            period = "off_peak"
        hourly.append({"hour": hour, "price_per_kwh": price, "period": period})
    return {"date": date, "currency": "USD", "hourly_prices": hourly}


@tool
def get_energy_usage(device_type: str = "all") -> dict[str, Any]:
    """Query historical energy usage for a device or all devices."""

    _ensure_data_ready()
    if device_type == "all":
        rows = query_rows(
            """
            SELECT device_type, ROUND(SUM(consumption_kwh), 3) AS total_kwh,
                   ROUND(SUM(cost_usd), 3) AS total_cost
            FROM EnergyUsage GROUP BY device_type ORDER BY total_cost DESC
            """
        )
    else:
        rows = query_rows(
            """
            SELECT device_type, STRFTIME('%H', timestamp) AS hour,
                   ROUND(AVG(consumption_kwh), 3) AS avg_kwh,
                   ROUND(AVG(cost_usd), 3) AS avg_cost
            FROM EnergyUsage WHERE device_type = ?
            GROUP BY device_type, hour ORDER BY hour
            """,
            (device_type,),
        )
    return {"device_type": device_type, "rows": rows}


@tool
def get_solar_generation(date: str = "recent") -> dict[str, Any]:
    """Query historical solar generation patterns."""

    _ensure_data_ready()
    rows = query_rows(
        """
        SELECT STRFTIME('%H', timestamp) AS hour,
               ROUND(AVG(generation_kwh), 3) AS avg_generation_kwh,
               GROUP_CONCAT(DISTINCT weather_condition) AS weather_conditions
        FROM SolarGeneration
        GROUP BY hour ORDER BY hour
        """
    )
    best = sorted(rows, key=lambda row: row["avg_generation_kwh"], reverse=True)[:5]
    return {"date": date, "hourly_generation": rows, "best_solar_hours": best}


def _load_or_create_collection():
    import chromadb

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection("energy_tips")

    if collection.count() > 0:
        sample = collection.get(limit=1)
        metadata = (sample.get("metadatas") or [{}])[0] or {}
        if "chunk_index" not in metadata:
            client.delete_collection("energy_tips")
            collection = client.get_or_create_collection("energy_tips")

    if collection.count() == 0:
        ids, documents, metadatas, embeddings = [], [], [], []
        for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            chunks = _split_text_into_chunks(text)
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{path.stem}-{chunk_index}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": path.name,
                        "topic": path.stem,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                    }
                )
                embeddings.append(_simple_embedding(chunk))
        if documents:
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return collection


@tool
def search_energy_tips(query: str) -> dict[str, Any]:
    """Search the ChromaDB energy-saving knowledge base.

    Args:
        query: Energy optimization question or topic.
    """

    collection = _load_or_create_collection()
    result = collection.query(query_embeddings=[_simple_embedding(query)], n_results=3)
    hits = []
    for idx, doc_id in enumerate(result.get("ids", [[]])[0]):
        hits.append(
            {
                "id": doc_id,
                "document": result.get("documents", [[]])[0][idx],
                "metadata": result.get("metadatas", [[]])[0][idx],
                "distance": result.get("distances", [[]])[0][idx] if result.get("distances") else None,
            }
        )
    return {"query": query, "results": hits}


@tool
def calculate_savings(current_hour: int, optimized_hour: int, kwh: float) -> dict[str, Any]:
    """Calculate cost savings from moving energy use between hours."""

    prices = get_electricity_prices.invoke("tomorrow")["hourly_prices"]
    current_price = prices[current_hour]["price_per_kwh"]
    optimized_price = prices[optimized_hour]["price_per_kwh"]
    savings = round((current_price - optimized_price) * kwh, 2)
    return {
        "current_hour": current_hour,
        "optimized_hour": optimized_hour,
        "kwh": kwh,
        "current_cost": round(current_price * kwh, 2),
        "optimized_cost": round(optimized_price * kwh, 2),
        "estimated_savings": savings,
        "co2_kg_avoided": round(max(savings, 0) * 2.1, 2),
    }


def get_all_tools():
    return [
        get_weather_forecast,
        get_electricity_prices,
        get_energy_usage,
        get_solar_generation,
        search_energy_tips,
        calculate_savings,
    ]

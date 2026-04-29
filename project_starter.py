import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# ----------------------------------------------------------------------
# Model + env setup
# ----------------------------------------------------------------------
import json
import re
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

dotenv.load_dotenv()

# OpenAI-compatible model. The Udacity Vocareum proxy is the canonical
# endpoint at submission time (https://openai.vocareum.com/v1 with key
# UDACITY_OPENAI_API_KEY). For local development this auto-falls-back
# to whichever OpenAI-compatible provider has a key in the env.
def _select_provider():
    if os.environ.get("UDACITY_OPENAI_API_KEY"):
        return ("vocareum",
                os.environ["UDACITY_OPENAI_API_KEY"],
                os.environ.get("UDACITY_OPENAI_API_BASE", "https://openai.vocareum.com/v1"),
                os.environ.get("UDACITY_OPENAI_MODEL", "gpt-4o-mini"))
    if os.environ.get("DEEPSEEK_API_KEY"):
        return ("deepseek",
                os.environ["DEEPSEEK_API_KEY"],
                "https://api.deepseek.com/v1",
                "deepseek-chat")
    if os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"):
        return ("kimi",
                os.environ.get("MOONSHOT_API_KEY") or os.environ["KIMI_API_KEY"],
                "https://api.moonshot.cn/v1",
                "moonshot-v1-32k")
    if os.environ.get("ZHIPU_API_KEY"):
        return ("glm",
                os.environ["ZHIPU_API_KEY"],
                "https://open.bigmodel.cn/api/paas/v4",
                "glm-4-plus")
    raise RuntimeError(
        "No OpenAI-compatible API key found. Set UDACITY_OPENAI_API_KEY (Vocareum) "
        "or one of: DEEPSEEK_API_KEY / MOONSHOT_API_KEY / ZHIPU_API_KEY."
    )

_provider, _API_KEY, _API_BASE, _MODEL_ID = _select_provider()
print(f"[model] provider={_provider} model={_MODEL_ID} base={_API_BASE}")
model = OpenAIServerModel(model_id=_MODEL_ID, api_base=_API_BASE, api_key=_API_KEY)

# ----------------------------------------------------------------------
# Tools — every public helper is wrapped here so all 7 functions are
# exercised across the agent system.
# ----------------------------------------------------------------------

# Bulk discount tiers used by the QuotingAgent. Larger orders get steeper cuts.
DISCOUNT_TIERS = [(5000, 0.15), (1000, 0.10), (500, 0.05)]


def _bulk_discount(qty: int) -> float:
    for threshold, pct in DISCOUNT_TIERS:
        if qty >= threshold:
            return pct
    return 0.0


# ---- Inventory tools ------------------------------------------------

@tool
def tool_get_all_inventory(as_of_date: str) -> str:
    """Return a JSON object mapping every in-stock item to its current quantity.

    Args:
        as_of_date: ISO date YYYY-MM-DD, the cutoff for the snapshot.
    """
    inv = get_all_inventory(as_of_date)
    inv_int = {k: int(v) for k, v in inv.items()}
    return json.dumps(inv_int)


@tool
def tool_get_stock_level(item_name: str, as_of_date: str) -> str:
    """Return the current stock for a single item as a JSON object.

    Args:
        item_name: Exact item name from the catalog (case-sensitive).
        as_of_date: ISO date YYYY-MM-DD.
    """
    df = get_stock_level(item_name, as_of_date)
    if df.empty:
        return json.dumps({"item_name": item_name, "current_stock": 0})
    return json.dumps({
        "item_name": str(df.iloc[0]["item_name"]),
        "current_stock": int(df.iloc[0]["current_stock"]),
    })


@tool
def tool_check_supplier_delivery(quantity: int, request_date: str) -> str:
    """Estimate the supplier delivery date for a stock-replenishment order.

    Args:
        quantity: Number of units to be shipped from the supplier.
        request_date: ISO date YYYY-MM-DD when the reorder is placed.
    """
    eta = get_supplier_delivery_date(request_date, quantity)
    return json.dumps({"reorder_date": request_date, "eta": eta, "quantity": int(quantity)})


@tool
def tool_record_stock_purchase(item_name: str, quantity: int, total_price: float, date: str) -> str:
    """Record a 'stock_orders' transaction (replenishing inventory from a supplier).

    Args:
        item_name: Catalog item name.
        quantity: Units purchased from the supplier.
        total_price: Total cost paid to the supplier.
        date: ISO date YYYY-MM-DD of the purchase.
    """
    tx_id = create_transaction(item_name, "stock_orders", int(quantity), float(total_price), date)
    return json.dumps({"transaction_id": tx_id, "kind": "stock_orders",
                       "item_name": item_name, "quantity": int(quantity),
                       "total_price": float(total_price), "date": date})


# ---- Quoting tools --------------------------------------------------

@tool
def tool_search_quote_history(search_terms: list, limit: int = 5) -> str:
    """Search the historical quotes for similar past pricing.

    Args:
        search_terms: List of keywords (event type, item type, etc.) to match.
        limit: Max number of past quotes to return.
    """
    rows = search_quote_history(list(search_terms), limit=int(limit))
    # Strip non-JSON-friendly objects
    clean = [{k: (str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v)
              for k, v in r.items()} for r in rows]
    return json.dumps(clean)


@tool
def tool_compute_quote(item_name: str, quantity: int, unit_price: float) -> str:
    """Compute the customer-facing quote line for an item using bulk-discount tiers.

    Args:
        item_name: Catalog item name.
        quantity: Quantity ordered by the customer.
        unit_price: Catalog unit price for the item.
    """
    discount = _bulk_discount(int(quantity))
    line_total = float(unit_price) * int(quantity) * (1.0 - discount)
    return json.dumps({
        "item_name": item_name,
        "quantity": int(quantity),
        "unit_price": float(unit_price),
        "discount_pct": round(discount * 100, 1),
        "line_total": round(line_total, 2),
    })


# ---- Sales tools ----------------------------------------------------

@tool
def tool_record_sale(item_name: str, quantity: int, total_price: float, date: str) -> str:
    """Record a 'sales' transaction (a fulfilled customer order line).

    Args:
        item_name: Catalog item name.
        quantity: Units sold.
        total_price: Final price paid by the customer for this line.
        date: ISO date YYYY-MM-DD of the sale.
    """
    tx_id = create_transaction(item_name, "sales", int(quantity), float(total_price), date)
    return json.dumps({"transaction_id": tx_id, "kind": "sales",
                       "item_name": item_name, "quantity": int(quantity),
                       "total_price": float(total_price), "date": date})


@tool
def tool_get_cash_balance(as_of_date: str) -> str:
    """Return the company's cash balance as of a date.

    Args:
        as_of_date: ISO date YYYY-MM-DD.
    """
    return json.dumps({"as_of_date": as_of_date, "cash_balance": float(get_cash_balance(as_of_date))})


@tool
def tool_generate_financial_report(as_of_date: str) -> str:
    """Return a finance snapshot {cash, inventory_value, total_assets} as of a date.

    Args:
        as_of_date: ISO date YYYY-MM-DD.
    """
    rep = generate_financial_report(as_of_date)
    return json.dumps({
        "cash_balance": float(rep["cash_balance"]),
        "inventory_value": float(rep["inventory_value"]),
        "total_assets": float(rep["total_assets"]),
    })


# ----------------------------------------------------------------------
# Worker agents (3) + Orchestrator (1) = 4 total, well under the 5 cap.
# ----------------------------------------------------------------------

inventory_agent = ToolCallingAgent(
    tools=[tool_get_all_inventory, tool_get_stock_level,
           tool_check_supplier_delivery, tool_record_stock_purchase],
    model=model,
    max_steps=8,
    name="inventory_agent",
    description=(
        "Checks stock levels and decides reorders. Inputs: a list of "
        "(item_name, quantity_needed) pairs and a date. For each item, "
        "looks up current stock; if stock < quantity_needed, places a "
        "stock-replenishment transaction sized to fill the gap and reports "
        "the supplier ETA. Returns a JSON list of per-item availability "
        "verdicts (FULFILLABLE / RESTOCKED / UNFULFILLABLE)."
    ),
    instructions=(
        "You are the inventory specialist. Given items and a date, call "
        "tool_get_stock_level for each item. If current_stock >= quantity, "
        "verdict is FULFILLABLE. Otherwise compute the shortfall and call "
        "tool_check_supplier_delivery to see if a restock arrives by the "
        "delivery deadline; if it does, call tool_record_stock_purchase "
        "(price = qty * unit_price, treat unit_price=1.0 if unknown) and "
        "mark RESTOCKED. If supplier ETA misses the deadline mark "
        "UNFULFILLABLE with a short reason. Return a single JSON array."
    ),
)

quoting_agent = ToolCallingAgent(
    tools=[tool_search_quote_history, tool_compute_quote, tool_check_supplier_delivery],
    model=model,
    max_steps=8,
    name="quoting_agent",
    description=(
        "Generates a price quote per line item, applying bulk discounts "
        "(5%/10%/15% at 500/1000/5000 units) and consulting historical "
        "quotes. Returns a JSON list with line_total and an explanation."
    ),
    instructions=(
        "Build a quote by calling tool_compute_quote for each item. Always "
        "call tool_search_quote_history once with relevant keywords (event "
        "type, item type) so prior pricing informs the explanation. Return "
        "a JSON object with 'lines' (list) and 'grand_total' (float)."
    ),
)

sales_agent = ToolCallingAgent(
    tools=[tool_record_sale, tool_get_cash_balance, tool_generate_financial_report],
    model=model,
    max_steps=6,
    name="sales_agent",
    description=(
        "Finalizes a sale by recording 'sales' transactions for each "
        "fulfillable line item, then returns a snapshot of post-sale cash "
        "balance and inventory value."
    ),
    instructions=(
        "For every line in the input, call tool_record_sale. Then call "
        "tool_generate_financial_report once and return JSON with the "
        "transaction ids and the finance snapshot."
    ),
)

orchestrator = ToolCallingAgent(
    tools=[tool_get_all_inventory, tool_get_cash_balance],
    model=model,
    max_steps=12,
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    instructions=(
        "You are the Munder Difflin sales orchestrator. For each customer "
        "request:\n"
        "1. Parse the free-text request into a list of (item_name, "
        "quantity) pairs and the delivery deadline. Match item names to "
        "the catalog as closely as possible (e.g. 'A4 glossy paper' -> "
        "'Glossy paper' or 'A4 paper' depending on context).\n"
        "2. Call inventory_agent with the parsed items + delivery date.\n"
        "3. For items marked FULFILLABLE or RESTOCKED, call quoting_agent "
        "to get a price.\n"
        "4. Call sales_agent with the priced items to record the sale.\n"
        "5. Compose ONE customer-facing response: list each item with its "
        "price (and discount %), give the grand total, mention delivery "
        "ETA, and for any UNFULFILLABLE item briefly explain why. Do NOT "
        "expose internal cash balances or supplier names."
    ),
)


def sanitize_customer_response(text_: str) -> str:
    """Trim agent meta-output and keep only the customer-facing answer."""
    if text_ is None:
        return ""
    s = str(text_).strip()
    # Strip common smolagents prefixes
    s = re.sub(r"^(?:Final answer|FINAL ANSWER|Answer)[:\s]*", "", s, flags=re.IGNORECASE)
    # Drop internal numeric tells
    s = re.sub(r"cash[_ ]?balance[^\n]*", "", s, flags=re.IGNORECASE)
    return s.strip() or "Quote generated."


# ----------------------------------------------------------------------
# Run scenarios
# ----------------------------------------------------------------------

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Multi-agent system was instantiated at module import time
    # (orchestrator + 3 worker agents = 4 agents, all sharing one model).
    print(f"Starting state: cash=${current_cash:.2f}  inventory=${current_inventory:.2f}")

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        # ------------------------------------------------------------------
        # Orchestrator handles parsing -> inventory -> quote -> sale and
        # composes a single customer-facing response.
        # ------------------------------------------------------------------
        try:
            raw = orchestrator.run(request_with_date)
            response = sanitize_customer_response(raw)
        except Exception as exc:
            response = f"We could not process this request automatically ({type(exc).__name__}: {exc})."
            print(f"  ! orchestrator error: {exc}")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()

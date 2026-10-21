"""Transformer project — Text Translation + Sentiment Analysis on movie reviews.

Rubric tasks:
 1. Load 3 CSVs (English, French, Spanish)
 2. Translate fr -> en and sp -> en using MarianMT
 3. Merge into single dataframe
 4. Sentiment analysis (Positive/Negative) using HuggingFace pipeline
 5. Export result/reviews_with_sentiment.csv
"""
import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

os.makedirs("result", exist_ok=True)
device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
print(f"device: {device}")

# --- Step 1: Load CSVs ---
df_eng = pd.read_csv("data/movie_reviews_eng.csv")
df_fr = pd.read_csv("data/movie_reviews_fr.csv").rename(
    columns={"Titre": "Movie / TV Series", "Année": "Year", "Critiques": "Review"}
)
df_sp = pd.read_csv("data/movie_reviews_sp.csv").rename(
    columns={"Título": "Movie / TV Series", "Año": "Year", "Sinopsis": "Synopsis", "Críticas": "Review"}
)

# Add Original Language column (per rubric)
df_eng["Original Language"] = "en"
df_fr["Original Language"] = "fr"
df_sp["Original Language"] = "es"

print(f"eng: {len(df_eng)} fr: {len(df_fr)} sp: {len(df_sp)}")


# --- Step 2: Translation ---
def translate_column(texts, src_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
    print(f"loading {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = []
    for text in texts:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated.append(out)
    return translated


print("\n--- translating French ---")
df_fr["Review"] = translate_column(df_fr["Review"].tolist(), "fr")
df_fr["Synopsis"] = translate_column(df_fr["Synopsis"].tolist(), "fr")

print("\n--- translating Spanish ---")
df_sp["Review"] = translate_column(df_sp["Review"].tolist(), "es")
df_sp["Synopsis"] = translate_column(df_sp["Synopsis"].tolist(), "es")

# --- Step 3: Merge ---
df = pd.concat([df_eng, df_fr, df_sp], ignore_index=True)
print(f"\nmerged df: {len(df)} rows")
print(df.columns.tolist())

# --- Step 4: Sentiment Analysis ---
print("\n--- sentiment analysis ---")
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU for stability on MPS
)


def analyze_sentiment(review: str) -> str:
    res = sentiment_classifier(review[:512])[0]  # truncate to model max
    return res["label"].capitalize()  # POSITIVE / NEGATIVE -> Positive / Negative


df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# --- Step 5: Export ---
out_path = "result/reviews_with_sentiment.csv"
df.to_csv(out_path, index=False)
print(f"\nExported -> {out_path}")
print(df[["Movie / TV Series", "Original Language", "Sentiment"]].to_string())
print(f"\nALL DONE — {len(df)} rows, {df['Sentiment'].value_counts().to_dict()}")

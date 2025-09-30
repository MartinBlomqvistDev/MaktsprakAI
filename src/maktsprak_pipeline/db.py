# =========================================================
# Fil: src/maktsprak_pipeline/db.py
# Syfte: Databasmodul för MaktsprakAI (Supabase via REST API + extern Parquet)
# Beroenden:
#   - supabase-py
#   - streamlit (valfritt, fallback till .env)
#   - logger
#   - tenacity
#   - pandas, pyarrow
# =========================================================

import os
import io
import random
from datetime import datetime
from tenacity import retry, wait_fixed, stop_after_attempt

import pandas as pd
import pyarrow.parquet as pq
import requests

from supabase import create_client
from .logger import get_logger
from .config import SUPABASE_URL, SUPABASE_KEY, PARQUET_URL


logger = get_logger()

# =========================================================
# Initiera Supabase-klient med stöd för lokal .env eller Streamlit
# =========================================================

try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False
    from dotenv import load_dotenv
    load_dotenv()

if USE_STREAMLIT:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# Läsfunktioner
# =========================================================

def fetch_speeches_count():
    """Returnerar antal rader i tabellen 'speeches'."""
    resp = supabase.table("speeches").select("protokoll_id", count="exact").execute()
    if resp.data is None:
        raise Exception(f"Supabase fetch failed: {resp}")
    return resp.count

def fetch_latest_speech_date():
    """Returnerar senaste protokoll_datum från tabellen 'speeches'."""
    resp = (
        supabase.table("speeches")
        .select("protokoll_datum")
        .order("protokoll_datum", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]["protokoll_datum"]

if USE_STREAMLIT:
    @st.cache_data(ttl=3600)
    def fetch_latest_speech_date_cached():
        return fetch_latest_speech_date()
else:
    fetch_latest_speech_date_cached = fetch_latest_speech_date

def fetch_random_speeches(limit: int = 5):
    """Hämtar slumpmässiga anföranden."""
    resp = supabase.table("speeches").select("*").execute()
    if not resp.data:
        return []
    data = resp.data
    random.shuffle(data)
    return data[:limit]

def fetch_speeches_historical(start_date="2015-01-01", end_date=None):
    """Hämtar alla historiska anföranden från Supabase mellan start_date och end_date."""
    dfs = []
    batch_size = 1000
    offset = 0

    query = supabase.table("speeches").select("id, text, parti, protokoll_datum")

    if start_date:
        query = query.gte("protokoll_datum", str(start_date))
    if end_date:
        query = query.lte("protokoll_datum", str(end_date))

    query = query.order("protokoll_datum", desc=False)

    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if resp.data:
            dfs.append(pd.DataFrame(resp.data))
            if len(resp.data) < batch_size:
                break
            offset += batch_size
        else:
            break

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["protokoll_datum"] = pd.to_datetime(df["protokoll_datum"], errors="coerce")
        df = df.dropna(subset=["protokoll_datum"])
        df = df.sort_values("protokoll_datum").reset_index(drop=True)
        return df

    return pd.DataFrame(columns=["id", "text", "parti", "protokoll_datum"])

# =========================================================
# Läs historisk Parquet externt
# =========================================================

def load_historical_parquet(parquet_url: str):
    """
    Läser Parquet med historisk data från en extern URL (t.ex. Google Drive).
    """
    if not parquet_url:
        logger.warning("Ingen Parquet URL definierad, returnerar tom DataFrame")
        return pd.DataFrame(columns=["id", "text", "parti", "protokoll_datum"])
    
    r = requests.get(parquet_url)
    r.raise_for_status()
    table = pq.read_table(io.BytesIO(r.content))
    df = table.to_pandas()
    df["protokoll_datum"] = pd.to_datetime(df["protokoll_datum"], errors="coerce")
    return df

# =========================================================
# Kombinera historisk Parquet + ny data
# =========================================================

def fetch_combined_speeches(start_date="2015-01-01", end_date=None):
    # Läs historisk Parquet via config
    df_historic = load_historical_parquet(PARQUET_URL)
    df_historic = df_historic[df_historic["protokoll_datum"] >= pd.to_datetime(start_date)]
    if end_date:
        df_historic = df_historic[df_historic["protokoll_datum"] <= pd.to_datetime(end_date)]
    
    # Hämta ny data från Supabase
    last_date = df_historic["protokoll_datum"].max() + pd.Timedelta(days=1) if not df_historic.empty else start_date
    df_new = fetch_speeches_historical(start_date=last_date, end_date=end_date)
    
    # Kombinera och ta bort dubbletter
    df_combined = pd.concat([df_historic, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"])
    df_combined = df_combined.sort_values("protokoll_datum").reset_index(drop=True)
    
    logger.info(f"Historisk data: {len(df_historic)}, Ny data: {len(df_new)}, Totalt: {len(df_combined)}")
    return df_combined



# =========================================================
# Skrivfunktioner
# =========================================================

def insert_speech(row: dict):
    """Infogar eller uppdaterar ett tal i tabellen 'speeches'."""
    resp = supabase.table("speeches").upsert(row, on_conflict="protokoll_id").execute()
    
    if resp.data is None:
        logger.error(f"Supabase upsert failed for row: {row}")
        raise Exception(f"Supabase upsert failed: {resp}")
    
    if resp.data:
        logger.info(f"Tal bearbetat (upsert) i 'speeches': {row.get('protokoll_id', 'no-id')}")
        return resp.data
    else:
        logger.warning(f"Upsert returnerade ingen data för {row.get('protokoll_id')}")
        return None

def insert_tweet(row: dict):
    """Infogar en ny tweet, undviker dubbletter baserat på tweet_id."""
    if "tweet_id" in row:
        existing = supabase.table("tweets").select("tweet_id").eq("tweet_id", row["tweet_id"]).execute()
        if existing.data:
            logger.warning(f"Tweet redan finns: tweet_id {row['tweet_id']} – hoppar över.")
            return existing.data

    resp = supabase.table("tweets").insert(row).execute()
    if resp.data is None:
        raise Exception(f"Supabase insert tweet failed: {resp}")
    logger.info("Ny tweet infogad i 'tweets'")
    return resp.data

# =========================================================
# Tabellhantering
# =========================================================

def create_all_tables():
    logger.warning(
        "Tabeller skapas och hanteras direkt i Supabase. "
        "Använd Supabase Dashboard eller SQL-skript för schemaändringar."
    )

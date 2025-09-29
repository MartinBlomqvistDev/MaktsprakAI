# =========================================================
# Fil: src/maktsprak_pipeline/db.py
# Syfte: Databasmodul för MaktspråkAI (Supabase via REST API)
# Beroenden:
#   - supabase-py
#   - streamlit
#   - logger
#   - tenacity
# =========================================================

from supabase import create_client
import streamlit as st
from .logger import get_logger
import pandas as pd
import random
from tenacity import retry, wait_fixed, stop_after_attempt
from datetime import datetime

logger = get_logger()

# -----------------------------
# Initiera Supabase-klienten
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Läsfunktioner
# -----------------------------
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

@st.cache_data(ttl=3600)
def fetch_latest_speech_date_cached():
    """Returnerar senaste protokoll_datum från tabellen 'speeches' med cache."""
    return fetch_latest_speech_date() 

def fetch_random_speeches(limit: int = 5):
    """Hämtar slumpmässiga anföranden."""
    resp = supabase.table("speeches").select("*").execute()
    if not resp.data:
        return []
    data = resp.data
    random.shuffle(data)
    return data[:limit]

@st.cache_data(ttl=1800)
def fetch_speeches_historical(start_date, end_date):
    """
    Returnerar DataFrame med text och parti för en viss period.
    Hämtar i batchar per år för stora datamängder.
    """
    dfs = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(
            current_start.replace(year=current_start.year + 1) - pd.Timedelta(days=1),
            end_date
        )
        resp = (
            supabase.table("speeches")
            .select("text, parti, protokoll_datum")
            .gte("protokoll_datum", current_start.isoformat())
            .lte("protokoll_datum", current_end.isoformat())
            .limit(None)  # Viktigt för att få alla rader
            .execute()
        )
        if resp.data:
            dfs.append(pd.DataFrame(resp.data))
        current_start = current_end + pd.Timedelta(days=1)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df['protokoll_datum'] = pd.to_datetime(df['protokoll_datum'], errors='coerce')
        return df
    return pd.DataFrame(columns=["text", "parti", "protokoll_datum"])

# -----------------------------
# Skrivfunktioner
# -----------------------------
def insert_speech(row: dict):
    """
    Infogar eller uppdaterar ett tal i tabellen 'speeches'.
    Använder upsert för att hantera dubbletter baserat på 'protokoll_id'.
    """
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

# -----------------------------
# Tabellhantering
# -----------------------------
def create_all_tables():
    logger.warning(
        "Tabeller skapas och hanteras direkt i Supabase. "
        "Använd Supabase Dashboard eller SQL-skript för schemaändringar."
    )

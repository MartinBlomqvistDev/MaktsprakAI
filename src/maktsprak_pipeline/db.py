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

def fetch_random_speeches(limit: int = 5):
    """Hämtar slumpmässiga anföranden."""
    resp = supabase.table("speeches").select("*").execute()
    if not resp.data:
        return []
    data = resp.data
    random.shuffle(data)
    return data[:limit]

@st.cache_data(ttl=1800)
def fetch_speeches_in_period(start_date, end_date):
    """Returnerar DataFrame med text och parti för en viss period."""
    resp = (
        supabase.table("speeches")
        .select("text, parti, protokoll_datum")
        .gte("protokoll_datum", str(start_date))
        .lte("protokoll_datum", str(end_date))
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)

# Retry-decorator för temporära nätverksproblem
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def safe_fetch_speeches_in_period(start_date, end_date):
    return fetch_speeches_in_period(start_date, end_date)

# -----------------------------
# Skrivfunktioner
# -----------------------------
def insert_speech(row: dict):
    """
    Infogar ett nytt tal i tabellen 'speeches', undviker dubbletter.
    Dubblettkontroll baseras på 'protokoll_id' om finns, annars 'protokoll_datum + parti'.
    """
    # --- Kontroll mot protokoll_id ---
    if "protokoll_id" in row:
        existing = (
            supabase.table("speeches")
            .select("protokoll_id")
            .eq("protokoll_id", row["protokoll_id"])
            .execute()
        )
        if existing.data:
            logger.warning(f"Tal redan finns: protokoll_id {row['protokoll_id']} – hoppar över.")
            return existing.data

    # --- Fallback: kontroll mot datum + parti ---
    existing = (
        supabase.table("speeches")
        .select("protokoll_id")
        .eq("protokoll_datum", row["protokoll_datum"])
        .eq("parti", row["parti"])
        .execute()
    )
    if existing.data:
        logger.warning(f"Tal redan finns för parti {row['parti']} på datum {row['protokoll_datum']} – hoppar över.")
        return existing.data

    # --- Insert om ingen dubblett ---
    resp = supabase.table("speeches").insert(row).execute()
    if resp.data is None:
        raise Exception(f"Supabase insert failed: {resp}")
    logger.info(f"Nytt tal infogat i 'speeches': {row.get('protokoll_id', 'no-id')}")
    return resp.data

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

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
def fetch_speeches_historical(start_date="2015-01-01", end_date=None):
    """
    Hämtar alla historiska anföranden från 2015-01-01 och framåt,
    eller inom ett valfritt datumintervall.
    """
    dfs = []
    batch_size = 1000
    offset = 0

    query = supabase.table("speeches").select("text, parti, protokoll_datum")

    # Filtrera på datum
    if start_date:
        query = query.gte("protokoll_datum", str(start_date))
    if end_date:
        query = query.lte("protokoll_datum", str(end_date))

    # Rätt param: desc=False betyder stigande ordning
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

        # Säkerställ rätt datumformat
        df["protokoll_datum"] = pd.to_datetime(df["protokoll_datum"], errors="coerce")

        # Ta bort rader utan giltigt datum
        df = df.dropna(subset=["protokoll_datum"])

        # Sortera en extra gång i Pandas
        df = df.sort_values("protokoll_datum").reset_index(drop=True)

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

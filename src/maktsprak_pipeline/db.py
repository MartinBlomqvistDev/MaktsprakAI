# =========================================================
# Fil: src/maktsprak_pipeline/db.py
# Syfte: Databasmodul för MaktspråkAI (Supabase via REST API)
# Beroenden:
#   - supabase-py
#   - streamlit
#   - logger
# =========================================================

from supabase import create_client
import streamlit as st
from .logger import get_logger

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
    resp = supabase.table("speeches").select("id", count="exact").execute()
    if resp.error:
        raise Exception(f"Supabase error (count): {resp.error}")
    return resp.count

def fetch_latest_speech_date():
    """Returnerar senaste protokoll_datum från tabellen 'speeches'."""
    resp = supabase.table("speeches").select("protokoll_datum").order("protokoll_datum", desc=True).limit(1).execute()
    if resp.error:
        raise Exception(f"Supabase error (latest date): {resp.error}")
    return resp.data[0]["protokoll_datum"] if resp.data else None

def fetch_random_speeches(limit: int = 5):
    """Hämtar slumpmässiga anföranden (kräver att RLS tillåter det)."""
    resp = supabase.table("speeches").select("*").limit(limit).execute()
    if resp.error:
        raise Exception(f"Supabase error (random speeches): {resp.error}")
    return resp.data

# -----------------------------
# Skrivfunktioner
# -----------------------------
def insert_speech(row: dict):
    """Infogar ett nytt tal i tabellen 'speeches'."""
    resp = supabase.table("speeches").insert(row).execute()
    if resp.error:
        raise Exception(f"Supabase error (insert): {resp.error}")
    logger.info("Nytt tal infogat i 'speeches'")
    return resp.data

def insert_tweet(row: dict):
    """Infogar en ny tweet i tabellen 'tweets'."""
    resp = supabase.table("tweets").insert(row).execute()
    if resp.error:
        raise Exception(f"Supabase error (insert tweet): {resp.error}")
    logger.info("Ny tweet infogad i 'tweets'")
    return resp.data

# -----------------------------
# Tabellhantering
# -----------------------------
# OBS: Tabellen måste skapas i Supabase-webbgränssnittet eller via SQL. 
# Funktionen nedan är mest till för att hålla koden konsekvent med tidigare version.
def create_all_tables():
    logger.warning("Tabeller skapas och hanteras direkt i Supabase. "
                   "Använd Supabase Dashboard eller SQL-skript för schemaändringar.")

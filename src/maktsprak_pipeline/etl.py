# src/maktsprak_pipeline/etl.py
import os
import re
import requests
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict
from pathlib import Path

import pdfplumber
import xml.etree.ElementTree as ET
from tqdm import tqdm

from .db import create_connection, create_all_tables
# FIX: Ta bort DB_PATH. create_connection ansluter nu utan argument.
from .config import RAW_DATA_PATH, X_API_KEY 
from .logger import get_logger

logger = get_logger()

# =========================================================
# Configuration
# =========================================================

PARTY_LEADERS_IDS = {
    "S": ["1587012835409788928"],
    "M": ["747426555417198592"],
    "V": ["282532238"],
    "L": ["455193032"],
    "KD": ["1407151866"],
    "C": ["232799403"],
    "MP": ["41214271", "370900852"],
    "SD": ["95972673"]
}

VALID_PARTIES = set(PARTY_LEADERS_IDS.keys())
MAX_TWEETS_PER_PARTY = 2
MONTHLY_TWEET_LIMIT = 100

# =========================================================
# Helper Functions
# =========================================================

def fetch_with_retry(url: str, headers: Dict, params: Dict = None, max_retries: int = 5):
    """
    Fetch data from API with retry on rate limit (429)
    """
    wait_time = 900  # 15 minutes
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        logger.error(f"API error {resp.status_code}: {resp.text}")
        return None
    logger.error(f"Failed after {max_retries} attempts: {url}")
    return None

# =========================================================
# Extract Tweets
# =========================================================

def extract_all_tweets() -> List[Dict]:
    headers = {"Authorization": f"Bearer {X_API_KEY}"}
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=7)

    # FIX: Ta bort DB_PATH från create_connection anropet
    with create_connection() as conn:
        cursor = conn.cursor()
        first_of_month = datetime(end_time.year, end_time.month, 1, tzinfo=timezone.utc)
        # FIX: Använd %s placeholder för PostgreSQL istället för SQLite ?
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE created_at >= %s", (first_of_month.isoformat(),))
        already_fetched = cursor.fetchone()[0]

    tweets_remaining = max(0, MONTHLY_TWEET_LIMIT - already_fetched)
    if tweets_remaining == 0:
        logger.info("Monthly tweet limit reached, skipping fetch.")
        return []
    
    tweets_per_party = defaultdict(list)

    for party in tqdm(PARTY_LEADERS_IDS.keys(), desc="Fetching tweets by party"):
        if tweets_remaining <= 0:
            break

        user_ids = PARTY_LEADERS_IDS[party]
        max_for_party = min(MAX_TWEETS_PER_PARTY, tweets_remaining)

        for user_id in user_ids:
            if len(tweets_per_party[party]) >= max_for_party:
                break

            params = {
                "max_results": 5,
                "start_time": start_time.isoformat().replace("+00:00", "Z"),
                "end_time": end_time.isoformat().replace("+00:00", "Z"),
                "tweet.fields": "id,text,created_at,lang"
            }
            url = f"https://api.twitter.com/2/users/{user_id}/tweets"
            resp = fetch_with_retry(url, headers, params)
            if not resp:
                continue

            tweets_data = sorted(resp.json().get("data", []), key=lambda x: x["created_at"], reverse=True)
            for t in tqdm(tweets_data, desc=f"Adding tweets for {party}", leave=False):
                if len(tweets_per_party[party]) >= max_for_party:
                    break
                tweets_per_party[party].append({
                    "id": t["id"],
                    "created_at": t["created_at"],
                    "username": user_id,
                    "lang": t.get("lang", "NA").upper(),
                    "text": t["text"].strip(),
                    "url": f"https://x.com/i/web/status/{t['id']}"
                })

        tweets_remaining -= len(tweets_per_party[party])
        logger.info(f"Fetched {len(tweets_per_party[party])} tweets for {party}, {tweets_remaining} remaining this month")

    all_tweets = [t for tweets in tweets_per_party.values() for t in tweets]
    logger.info(f"Total tweets fetched: {len(all_tweets)}")
    return all_tweets

# =========================================================
# Load Tweets
# =========================================================

def load_tweets(data: List[Dict]):
    create_all_tables()
    # FIX: Ta bort DB_PATH från create_connection anropet
    with create_connection() as conn:
        cursor = conn.cursor()
        for row in data:
            # FIX: Ändra placeholders från SQLite (?) till PostgreSQL (%s)
            cursor.execute("""
                INSERT INTO tweets (id, created_at, username, lang, text, url)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING; -- PostgreSQL motsvarighet till IGNORE
            """, (row["id"], row["created_at"], row["username"], row["lang"], row["text"], row["url"]))
        conn.commit()
    logger.info(f"Loaded {len(data)} tweets into database (duplicates ignored)")

# =========================================================
# Extract Riksdag Protocols
# =========================================================

def extract_riksdag_protocols(lookback_days: int = 7, max_back: int = 90) -> str:
    today = datetime.today()
    found_docs = False
    start_date = today # Initialize to avoid reference before assignment
    while not found_docs:
        start_date = today - timedelta(days=lookback_days)
        api_url = (
            "https://data.riksdagen.se/dokumentlista/"
            "?sok=&doktyp=prot"
            f"&from={start_date.strftime('%Y-%m-%d')}&tom={today.strftime('%Y-%m-%d')}"
            "&utformat=xml&sort=datum&sortorder=desc"
        )
        resp = requests.get(api_url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        doc_count = len(root.findall(".//dokument"))

        if doc_count > 0:
            found_docs = True
            logger.info(f"Found {doc_count} protocols from {start_date.date()} to {today.date()}")
        else:
            logger.info(f"No protocols found, backing off 1 week")
            lookback_days += 7
            if lookback_days > max_back:
                logger.warning("No protocols found in 3 months, aborting")
                return None # Return None if no docs are found

        filename = os.path.join(
            RAW_DATA_PATH,
            f"protocols_{start_date.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}.xml"
        )
        with open(filename, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved protocol list: {filename}")
        return filename

# =========================================================
# Transform Riksdag
# =========================================================

def transform_riksdag(xml_file: str) -> list[dict]:
    if not xml_file:
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    # *** KORRIGERING: Skärpt regex som bara accepterar 1-2 versaler för partiet ***
    regex_speech = re.compile(r"Anf\.\s+\d+\s+(.*?)\s+\(([A-ZÅÄÖ]{1,2})\):(.*?)(?=Anf\.|\Z)", re.S)

    documents = root.findall(".//dokument")
    for doc in tqdm(documents, desc="Processing protocols"):
        protocol_id = doc.findtext("dok_id")
        protocol_date = doc.findtext("datum")
        file_url = doc.findtext("filbilaga/fil/url")
        if not file_url:
            continue

        pdf_path = Path(RAW_DATA_PATH) / f"{protocol_id}.pdf"
        if not pdf_path.exists():
            resp_pdf = requests.get(file_url)
            resp_pdf.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(resp_pdf.content)

        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=2) or ""
                    page_text = re.sub(r'-\n', '', page_text)
                    page_text = re.sub(r'\n', ' ', page_text)
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = re.sub(r'\s{2,}', ' ', page_text) # Extra rensning av mellanslag
                    text += page_text + " "

            # Gruppera anföranden för att slå ihop flera anföranden av samma person
            grouped = defaultdict(list)
            for speaker, party, speech_text in regex_speech.findall(text):
                if party in VALID_PARTIES:
                    speech_text_cleaned = speech_text.strip()
                    if speech_text_cleaned:
                        grouped[(speaker.strip(), party)].append(speech_text_cleaned)

            for idx, ((speaker, party), speeches) in enumerate(grouped.items(), start=1):
                speech_id = f"{protocol_id}_{idx}"
                combined_text = "\n\n".join(speeches)
                data.append({
                    "id": speech_id,
                    "protocol_id": protocol_id,
                    "protocol_date": protocol_date,
                    "speaker": speaker,
                    "party": party,
                    "text": combined_text,
                    "file_url": file_url
                })

        except Exception as e:
            logger.warning(f"Failed to read PDF {protocol_id}: {e}")

    logger.info(f"Total speeches transformed: {len(data)}")
    return data

# =========================================================
# Load Riksdag
# =========================================================

def load_riksdag(data: List[Dict]):
    # FIX: Ta bort DB_PATH från create_connection anropet
    with create_connection() as conn:
        cursor = conn.cursor()
        for row in data:
            # FIX: Ändra placeholders från SQLite (?) till PostgreSQL (%s)
            cursor.execute("""
                INSERT INTO speeches 
                (id, protokoll_id, protokoll_datum, talare, parti, text, fil_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING; -- PostgreSQL motsvarighet till IGNORE
            """, (
                row["id"],
                row["protocol_id"],
                row["protocol_date"],
                row["speaker"],
                row["party"],
                row["text"],
                row["file_url"]
            ))
        conn.commit()
    logger.info(f"Loaded {len(data)} speeches into database")

# =========================================================
# Run ETL
# =========================================================

def run_etl():
    logger.info("===== Starting MaktsprakAI ETL =====")
    
    # FIX: Inaktivera ETL för att fokusera på stabil deployment med förladdad data
    logger.info("ETL är inaktiverad. Appen körs med förladdad data.")
    logger.info("===== ETL Finished =====")


def clean_invalid_parties():
    # FIX: Anropa create_connection utan argument (ansluter till molnet)
    try:
        with create_connection() as conn: 
            cursor = conn.cursor()
            # FIX: Använd PostgreSQL-placeholders (%s) istället för SQLite (?)
            cursor.execute(f"""
                DELETE FROM speeches
                WHERE parti NOT IN ({','.join(['%s']*len(VALID_PARTIES))}) OR LENGTH(parti) > 2
            """, tuple(VALID_PARTIES))
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                logger.info(f"Safety clean: Deleted {deleted} invalid speeches from DB.")
    except Exception as e:
        # Lägg till en fallback för att hantera att vi nu anropar den här funktionen i main.py:
        logger.warning(f"Skipping party cleanup: Connection failed or cleanup query needs update: {e}")

if __name__ == "__main__":
    try:
        run_etl()
        print("ETL completed successfully!")
    except Exception as e:
        print(f"ETL failed: {e}")
        logger.exception("ETL aborted due to error")
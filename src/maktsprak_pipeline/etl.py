# =========================================================
# Fil: src/maktsprak_pipeline/etl.py
# Syfte: ETL-flöde för MaktsprakAI
#   - Riksdag (tunga PDF/protokoll)
#   - Tweets (små batcher, senaste veckan)
# Beroenden:
#   - pandas, pdfplumber, requests, tqdm, xml.etree.ElementTree, python-dotenv
# =========================================================

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
from dotenv import load_dotenv

# =========================================================
# Läs .env lokalt
# =========================================================
load_dotenv()  # Läser SUPABASE_URL, SUPABASE_KEY, TWITTER_BEARER_TOKEN etc

from .db import create_connection, create_all_tables
from .config import RAW_DATA_PATH, X_API_KEY
from .logger import get_logger

logger = get_logger()

# =========================================================
# Konfiguration
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
# Hjälpfunktioner
# =========================================================

def fetch_with_retry(url: str, headers: Dict, params: Dict = None, max_retries: int = 5):
    """ Hämtar data från API med retry vid rate limit (429) """
    wait_time = 900  # 15 minuter
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            logger.warning(f"Rate limit hit, väntar {wait_time}s (försök {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        logger.error(f"API error {resp.status_code}: {resp.text}")
        return None
    logger.error(f"Failed efter {max_retries} försök: {url}")
    return None

# =========================================================
# Riksdag (Extract → Transform → Load)
# =========================================================

def extract_riksdag_protocols(lookback_days: int = 7, max_back: int = 90) -> str:
    today = datetime.today()
    found_docs = False
    start_date = today
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
            logger.info(f"Hittade {doc_count} protokoll från {start_date.date()} till {today.date()}")
        else:
            logger.info("Inga protokoll funna, backar 1 vecka")
            lookback_days += 7
            if lookback_days > max_back:
                logger.warning("Inga protokoll på 3 månader, avbryter")
                return None

        filename = os.path.join(
            RAW_DATA_PATH,
            f"protocols_{start_date.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}.xml"
        )
        with open(filename, "wb") as f:
            f.write(resp.content)
        logger.info(f"Sparad protokollslista: {filename}")
        return filename

def transform_riksdag(xml_file: str) -> List[Dict]:
    if not xml_file:
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

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
                    page_text = re.sub(r'\s{2,}', ' ', page_text)
                    text += page_text + " "

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
            logger.warning(f"Misslyckades läsa PDF {protocol_id}: {e}")

    logger.info(f"Totalt tal transformerade: {len(data)}")
    return data

def load_riksdag(data: List[Dict]):
    with create_connection() as conn:
        cursor = conn.cursor()
        for row in data:
            cursor.execute("""
                INSERT INTO speeches 
                (id, protokoll_id, protokoll_datum, talare, parti, text, fil_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
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
    logger.info(f"{len(data)} riksdagstal laddade till DB")

# =========================================================
# Tweets (Extract → Load)
# =========================================================

def extract_all_tweets() -> List[Dict]:
    headers = {"Authorization": f"Bearer {X_API_KEY}"}
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(days=7)

    with create_connection() as conn:
        cursor = conn.cursor()
        first_of_month = datetime(end_time.year, end_time.month, 1, tzinfo=timezone.utc)
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE created_at >= %s", (first_of_month.isoformat(),))
        already_fetched = cursor.fetchone()[0]

    tweets_remaining = max(0, MONTHLY_TWEET_LIMIT - already_fetched)
    if tweets_remaining == 0:
        logger.info("Månadsgräns för tweets uppnådd, hoppar över fetch")
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
            for t in tweets_data:
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
        logger.info(f"{len(tweets_per_party[party])} tweets hämtade för {party}, {tweets_remaining} kvar denna månad")

    all_tweets = [t for tweets in tweets_per_party.values() for t in tweets]
    logger.info(f"Totalt tweets hämtade: {len(all_tweets)}")
    return all_tweets

def load_tweets(data: List[Dict]):
    create_all_tables()
    with create_connection() as conn:
        cursor = conn.cursor()
        for row in data:
            cursor.execute("""
                INSERT INTO tweets (id, created_at, username, lang, text, url)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (row["id"], row["created_at"], row["username"], row["lang"], row["text"], row["url"]))
        conn.commit()
    logger.info(f"{len(data)} tweets laddade till DB (dubbletter ignorerade)")

# =========================================================
# Run ETL
# =========================================================

def run_etl():
    logger.info("===== Startar MaktsprakAI ETL =====")

    xml_file = extract_riksdag_protocols()
    riksdag_data = transform_riksdag(xml_file)
    if riksdag_data:
        load_riksdag(riksdag_data)

    tweet_data = extract_all_tweets()
    if tweet_data:
        load_tweets(tweet_data)

    logger.info("===== ETL färdig =====")

# =========================================================
# Safeguard: Radera ogiltiga partier
# =========================================================

def clean_invalid_parties():
    try:
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                DELETE FROM speeches
                WHERE parti NOT IN ({','.join(['%s']*len(VALID_PARTIES))}) OR LENGTH(parti) > 2
            """, tuple(VALID_PARTIES))
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                logger.info(f"Safety clean: Raderade {deleted} ogiltiga speeches från DB.")
    except Exception as e:
        logger.warning(f"Party cleanup misslyckades: {e}")

# =========================================================
# Kör ETL manuellt
# =========================================================

if __name__ == "__main__":
    try:
        run_etl()
        print("ETL completed successfully!")
    except Exception as e:
        print(f"ETL failed: {e}")
        logger.exception("ETL aborted due to error")

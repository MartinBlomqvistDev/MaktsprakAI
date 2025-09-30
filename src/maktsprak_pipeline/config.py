# src/maktsprak_pipeline/config.py
# Konfigurationsvariabler för MaktsprakAI
# UPPDATERAD FÖR POSTGRESQL/SUPABASE + PARQUET

from dotenv import load_dotenv
import os

# Ladda .env (används lokalt för att läsa secrets, ignoreras av Streamlit Secrets)
load_dotenv()

# =========================================================
# Databas (PostgreSQL/Supabase)
# =========================================================
# OBS: Dessa variabler läser in värden från Streamlit Cloud Secrets (eller .env lokalt)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", 5432) 

# Supabase REST API (alternativ om du inte vill köra direkt mot Postgres)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# =========================================================
# Extern data / Parquet
# =========================================================
PARQUET_URL = os.getenv("PARQUET_URL")  # Google Drive eller annan extern länk

# =========================================================
# Lokala filer
# =========================================================
RAW_DATA_PATH = "data/raw/"                # Rådata (tweets, debattext)
PROCESSED_DATA_PATH = "data/processed/"    # Preprocessad text
TON_LEXICON_PATH = "data/processed/ton_lexicon_with_weights.csv"

# =========================================================
# Loggning
# =========================================================
LOG_LEVEL = "INFO"                         # Loggnivå

# =========================================================
# API-URL:er och Nycklar
# =========================================================
RIKSDAG_API_URL = "https://data.riksdagen.se/dokumentlista/"  # Riksdagens dokumentlista
X_API_URL = "https://api.twitter.com/2/tweets"                # X (Twitter) API

# API-nycklar (från .env / Secrets)
RIKSDAG_API_KEY = os.getenv("RIKSDAG_API_KEY")
X_API_KEY = os.getenv("TWITTER_BEARER_TOKEN")

# =========================================================
# AI-Modell / Hugging Face
# =========================================================
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "MartinBlomqvist/maktsprak_bert")

# =========================================================
# Träningskonfiguration (Används av train_party_model_db.py)
# =========================================================
TRAIN_MODEL_NAME = "KB/bert-base-swedish-cased"
TRAIN_MODEL_DIR = "data/models/party_classifier" 
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_EPOCHS = 20
TRAIN_MAX_LENGTH = 512
TRAIN_LEARNING_RATE = 3e-5
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_EARLY_STOPPING_PATIENCE = 5
TRAIN_LABEL_SMOOTHING = 0.05
BASE_BATCH_SIZE = 16
BASE_MAX_LR = 5e-5

# =========================================================
# Fil: scripts/create_historic_database.py
# Syfte: Skapa en initial Parquet-fil med HELA speeches-tabellen
#        och ladda upp den till Supabase Storage (bucket: Historical_data).
# =========================================================

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from supabase import create_client
from datetime import date
from dotenv import load_dotenv

# ---------------------------------------------------------
# Ladda miljövariabler (från .env)
# ---------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # används för SELECT (läsning)
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # används för STORAGE (skrivning)

if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Saknar SUPABASE_URL, SUPABASE_KEY eller SUPABASE_SERVICE_KEY i .env")

# Klient för att läsa från databasen
supabase_read = create_client(SUPABASE_URL, SUPABASE_KEY)

# Klient för att skriva till Storage
supabase_write = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------
# Hämta ALL data från Supabase (speeches)
# ---------------------------------------------------------
def fetch_all_speeches():
    batch_size = 1000
    offset = 0
    dfs = []

    query = supabase_read.table("speeches").select("*").order("protokoll_datum", desc=False)

    while True:
        resp = query.range(offset, offset + batch_size - 1).execute()
        if not resp.data:
            break
        dfs.append(pd.DataFrame(resp.data))
        print(f"Hämtat {len(resp.data)} rader (totalt {sum(len(df) for df in dfs)})")
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not dfs:
        raise RuntimeError("Inga data hämtades från Supabase.")

    df = pd.concat(dfs, ignore_index=True)
    df["protokoll_datum"] = pd.to_datetime(df["protokoll_datum"], errors="coerce")
    return df


# ---------------------------------------------------------
# Skriv Parquet lokalt
# ---------------------------------------------------------
def write_parquet(df: pd.DataFrame, out_dir="data/parquet"):
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    filename = f"speeches_historic_{today}.parquet"
    filepath = os.path.join(out_dir, filename)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression="snappy")

    print(f"Parquet skapad: {filepath}")
    return filepath, filename

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Hämtar ALL historisk data (speeches)...")
    df = fetch_all_speeches()

    print("Skriver Parquet...")
    local_path, filename = write_parquet(df)


    print("\nKLART! Lägg till denna rad i .env eller st.secrets:")
    print(f'PARQUET_HISTORIC_URL="{public_url}"')

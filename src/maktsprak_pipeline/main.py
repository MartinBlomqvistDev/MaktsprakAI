# src/maktsprak_pipeline/main.py
from .etl import run_etl
from .logger import get_logger

logger = get_logger()

def main():
    print("MaktsprakAI ETL startar…")
    try:
        run_etl()  # <-- Denna anropar nu den RIKTIGA run_etl() som du har fixat
    except Exception:
        logger.exception("ETL-körning misslyckades")
        print("Något gick fel, kolla loggen!")
    else:
        logger.info("ETL-körning klar")
        print("MaktsprakAI ETL färdig!")

if __name__ == "__main__":
    main()
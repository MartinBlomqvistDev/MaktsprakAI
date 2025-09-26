@echo off
REM Byt till projektmappen (roten)
cd /d C:\DS24\MaktsprakAI

REM Aktivera virtual environment
call .venv\Scripts\activate.bat

REM Kör ETL och logga output
python -m src.maktsprak_pipeline.etl >> etl.log 2>&1

REM Avaktivera virtual environment
deactivate

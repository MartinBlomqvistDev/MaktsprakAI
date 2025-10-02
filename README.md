# MaktspråkAI: Analys av Sveriges politiska språkbruk

**Projektets slutmål:** Att etablera en robust, end-to-end data pipeline för insamling och AI-analys av politisk text, deployerad i molnet.

## 1. Översikt och Huvudfunktionalitet

Detta projekt demonstrerar en detaljerad och systematisk förståelse för **Data Science** och **NLP** genom att hantera hela kedjan: ETL, molndatabaser, state-of-the-art AI-modellering och deployment av en storskalig data-applikation.

### Nyckelfunktioner
* **Datainsamling:** Automatisk hämtning av Riksdagsprotokoll och Twitterdata via API:er.
* **AI-Prediktion:** Använder en finjusterad svensk BERT-modell för att förutsäga ett partis tillhörighet baserat på text.
* **Retorikanalys:** Visuell jämförelse av partiernas språkbruk och ton över tid.
* **Hållbarhet:** Systemet är helt molnbaserat (Supabase, Hugging Face) för skalbarhet och enkel deployment.

---

## 2. Arkitektur & Filstruktur

Projektet använder en **modulär arkitektur** där all återanvändbar logik ligger i en kärnmodul (`src/maktsprak_pipeline/`) och körbara instanser ligger i projektets rot.


```bash
/MaktsprakAI
├── src/
│   └── maktsprak_pipeline/
│       ├── __init__.py       # Initierar paketet
│       ├── config.py         # Centraliserad konfiguration och Secrets-läsning
│       ├── db.py             # PostgreSQL/Supabase-anslutning
│       ├── etl.py            # Data-pipeline: Extract, Transform, Load
│       ├── nlp.py            # Text- och lingvistisk behandling
│       ├── model.py          # Hanterar AI-modellens I/O
│       └── logger.py         # Central loggning för pipeline och app
├── app/
│   └── streamlit_app.py      # Deployment-applikationen
├── scripts/
│   ├── train_party_model_db.py   # Tränar BERT-modellen
│   └── create_historic_database.py  # Skapar historisk Parquet-dump och laddar till Supabase
├── main.py                   # Huvudkörning för pipeline och tester
├── requirements.txt          # Python-beroenden för molnet
└── .env                      # Ignoreras av git (Secrets)
```
---


## 3. Körning och Installation

1. **Kloning:** `git clone [din_repo]`
2. **Installation:** `pip install -r requirements.txt`
3. **Secrets:** Sätt dina Supabase- och Twitter-variabler som **Secrets** i din deployment-miljö.
4. **Kör appen:** `streamlit run app/streamlit_app.py`

***

# 4. Appendix: Teknisk Rapportfördjupning (VG-Krav)

Detta avsnitt detaljerar de tekniska val och systematiska lösningar som ligger till grund för projektets VG-nivå.

### 4.1 Modellutveckling och Optimering (`train_party_model_db.py`)

Modellen finjusterades på en svensk-specifik BERT-modell med följande optimeringar för att demonstrera detaljerad förståelse för modern NLP:

| Teknik | VG-Förklaring |
| :--- | :--- |
| **Weighted Sampling** | Istället för att bara använda klassvikter i loss-funktionen, används **`WeightedRandomSampler`** [cite: train_party_model_db.py]. Detta säkerställer att minibatchen aktivt balanserar samples från minoritetsklasserna (t.ex., MP, C), vilket är mer effektivt för att uppnå hög precision på hela klassuppsättningen. |
| **Adversarial Training (FGM)** | Modellen utsätts för beräknade små störningar (brus) på embedding-nivå under träning. Denna process, känd som Fast Gradient Method, gör modellen **mer robust** och motståndskraftig mot små variationer i språket [cite: train_party_model_db.py]. |
| **OneCycleLR & AMP** | Använder en avancerad *Learning Rate* scheduler (cyklisk) för snabb konvergens kombinerat med **Mixed Precision (AMP)** för att halvera minnesförbrukningen och **öka träningshastigheten** på GPU [cite: train_party_model_db.py]. |
| **Resursladdning** | Modellen och lexikonet laddas direkt från **Hugging Face** via **`@st.cache_resource`** [cite: model.py, streamlit_app.py]. Detta är den moderna deployment-standarden som eliminerar lokala filberoenden. |

### 4.2 Databas- och Migreringslösningar (Hållbarhet)

Projektet löste problemet med en **417 MB stor databasfil** genom att migrera till molnet och utveckla en systematisk lösning för import:

| Tekniskt Problem | VG-Implementering |
| :--- | :--- |
| **Molnmigrering** | Val av **Supabase (PostgreSQL)**, en permanent molntjänst [cite: db.py]. All anslutningsinformation hanteras säkert via `psycopg2-binary`.|
| **Kringgå DB-importbegränsningar**| Datan delades upp i flera CSV-filer **under 100 MB** [cite: image_8848c3.png]. Tabellerna skapades initialt utan Primärnyckel, och den korrekta nyckeln sattes sedan via **`ALTER TABLE ADD PRIMARY KEY`** i SQL Editor *efter* att datan var inläst. |
| **Säkerhet & Konfiguration** | Alla känsliga credentials läses in via **`os.getenv()`** från Streamlit Cloud Secrets, vilket är den professionella standarden för att hantera secrets i molnmiljöer [cite: config.py].
# =========================================================
# Fil: streamlit_app.py
# Syfte: Interaktiv dashboard för MaktspråkAI
# =========================================================

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta, date
import random
import re
from streamlit_option_menu import option_menu

import feedparser
from bs4 import BeautifulSoup
import requests
from huggingface_hub import hf_hub_download

plt.rcParams['font.family'] = 'sans-serif'

# =====================
# Sökvägshantering
# =====================
proj_root = Path(__file__).parent.parent.resolve()
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

# =====================
# Importer
# =====================
from src.maktsprak_pipeline.db import (
    fetch_speeches_count,
    fetch_latest_speech_date_cached,
    fetch_random_speeches,
    fetch_speeches_in_period,
    insert_speech,
    insert_tweet
)
from src.maktsprak_pipeline.nlp import apply_ton_lexicon, combined_stopwords, clean_text
from src.maktsprak_pipeline.model import load_model_and_tokenizer, predict_party

# =====================
# App-inställningar
# =====================
st.set_page_config(
    page_title="MaktspråkAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# Konstanter
# =====================
PARTY_ORDER = ["V", "MP", "S", "C", "L", "KD", "M", "SD"] 
PAGE_OPTIONS = ["Om projektet", "Partiprediktion", "Språkbruk & Retorik", "Evaluering", "Historik"]

# =====================
# Helper-funktioner
# =====================
def preprocess_for_wordcloud(text_blob: str, min_length: int = 3) -> str:
    words = re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', text_blob).lower().split()
    filtered_words = [word for word in words if word not in combined_stopwords and len(word) >= min_length]
    return " ".join(filtered_words)

@st.cache_data(ttl=900)
def fetch_news(feed_url="http://www.svt.se/nyheter/inrikes/rss.xml"):
    feed = feedparser.parse(feed_url)
    news_items = []
    for entry in feed.entries[:5]:
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })
    return news_items

@st.cache_data(ttl=3600)
def get_full_article_text(url: str) -> str:
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        possible_containers = [
            soup.find(class_='c-article__content'),
            soup.find('article'),
            soup.find(class_='article-body'),
            soup.find(class_='entry-content'),
            soup.find(class_='td-post-content'),
            soup.find('main')
        ]
        main_content = next((c for c in possible_containers if c), None)
        if main_content:
            for unwanted in main_content.find_all(['figure', 'figcaption', 'script', 'aside', 'header', 'footer']):
                unwanted.decompose()
            paragraphs = main_content.find_all('p')
            full_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            return full_text.strip()
        return ""
    except Exception as e:
        print(f"Ett fel uppstod vid skrapning av {url}: {e}")
        return ""

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_party_articles(articles_per_party: int = 1):
    party_feeds = {
        "S": "https://via.tt.se/rss/releases/latest?publisherId=142377",
        "M": "https://moderaterna.se/feed/",
        "SD": "https://sd.se/feed/",
        "C": "https://via.tt.se/rss/releases/latest?publisherId=3237070",
        "V": "https://vansterpartiet.se/feed/",
        "KD": "https://via.tt.se/rss/releases/latest?publisherId=3236814",
        "L": "https://www.liberalerna.se/feed/",
        "MP": "https://via.tt.se/rss/releases/latest?publisherId=3237031"
    }
    
    all_valid_articles = []
    debug_log = []
    
    for party, url in party_feeds.items():
        found_for_party_count = 0
        entries = []
        try:
            feed = feedparser.parse(url)
            entries = feed.entries
            if not entries and party == "S":
                debug_log.append(f"INFO [S]: RSS tomt. Kör special-skrapa för S nyhetssida.")
                try:
                    s_url = "https://www.socialdemokraterna.se/nyheter/nyheter"
                    response = requests.get(s_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                    s_soup = BeautifulSoup(response.content, "html.parser")
                    article_cards = s_soup.find_all('a', class_='c-card')
                    for card in article_cards[:10]:
                        title = card.find('h3', class_='c-card__title')
                        if title and card.has_attr('href'):
                            full_link = card['href']
                            if full_link.startswith('/'):
                                full_link = "https://www.socialdemokraterna.se" + full_link
                            entries.append({"title": title.get_text(strip=True), "link": full_link})
                except Exception as e:
                    debug_log.append(f"CRITICAL [S]: Special-skrapan misslyckades: {e}")

            debug_log.append(f"INFO [{party}]: Hittade {len(entries)} inlägg att bearbeta.")
            
            for i, entry in enumerate(entries):
                if found_for_party_count >= articles_per_party:
                    break
                title = entry.get('title', "Titel saknas")
                link = entry.get('link', None)
                if not link: continue
                debug_log.append(f"  -> Försöker hämta artikel {i+1}: '{title}'")
                full_content = get_full_article_text(link)
                if not full_content or len(full_content) < 250:
                    debug_log.append(f"    - MISSLYCKADES: Skrapan hittade för lite text (<250 tecken).")
                    continue
                if is_unwanted_content(title, full_content):
                    debug_log.append(f"    - MISSLYCKADES: Innehållet flaggades som 'oönskat'.")
                    continue
                debug_log.append(f"    - OK: Artikeln godkändes.")
                found_for_party_count += 1
                all_valid_articles.append({ "title": title, "link": link, "content": full_content, "true_party": party })
        except Exception as e:
            debug_log.append(f"CRITICAL [{party}]: Ett allvarligt fel inträffade: {e}")
            
    random.shuffle(all_valid_articles)
    return {"articles": all_valid_articles, "log": debug_log}

def is_unwanted_content(title: str, content: str) -> bool:
    title_lower = title.lower()
    content_lower = content.lower()
    text_length = len(content)
    announcement_keywords = ["välkommen till", "bjuder in", "schema:", "anmälan", "plats:", "program:", "agenda:"]
    if any(k in content_lower for k in announcement_keywords):
        return True
    job_ad_keywords = ["jobba hos oss", "söker", "ansök", "kvalifikationer", "anställning", "rekryterar", "ledig tjänst"]
    if any(k in title_lower or k in content_lower[:500] for k in job_ad_keywords):
        return True
    weak_filter_keywords = ["video:", "live:", "se talet", "anförande", "pressträff", "intervju med", "frågestund", "turné", "besöker"]
    if any(k in title_lower for k in weak_filter_keywords) and text_length < 500:
        return True
    return False

# =====================
# Ladda modell, tokenizer och lexikon
# =====================
@st.cache_resource(show_spinner="Laddar AI-modell och lexikon...")
def load_all_resources():
    model, tokenizer = load_model_and_tokenizer() 
    lexicon_local_path = hf_hub_download(
        repo_id="MartinBlomqvist/maktsprak_bert",
        filename="politisk_ton_lexikon.csv",
        revision="main"
    )
    return model, tokenizer, Path(lexicon_local_path)

model, tokenizer, LEXICON_PATH = load_all_resources()

# =====================
# Gemensam och cachad funktion för all evaluering
# =====================
@st.cache_data(ttl=60)
def get_data_signature():
    count = fetch_speeches_count()
    latest_date = fetch_latest_speech_date_cached()
    return (count, latest_date)

@st.cache_data(show_spinner="Värmer upp AI-modellen...")
def run_live_evaluation(articles_per_party: int = 5):
    fetch_results = fetch_party_articles(articles_per_party=articles_per_party)
    articles_to_analyze = fetch_results.get("articles", [])
    
    if not articles_to_analyze:
        return pd.DataFrame(), 0.0, 0

    results = []
    for article in articles_to_analyze:
        cleaned_for_model = clean_text(article['content'])  # <-- Ändrat här
        party_probs = predict_party(model, tokenizer, [cleaned_for_model])
        predicted_party = max(party_probs[0].items(), key=lambda x: x[1])[0]
        results.append({
            "Titel": article['title'], 
            "Sant parti": article['true_party'], 
            "Modellens gissning": predicted_party,
            "Korrekt?": (article['true_party'] == predicted_party)
        })
    results_df = pd.DataFrame(results)
    total_count = len(results_df)
    correct_count = results_df["Korrekt?"].sum()
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

    return results_df, accuracy, total_count

# =====================
# Välkomstsida
# =====================
def welcome_page():
    st.title("MaktspråkAI Dashboard")
    st.markdown("En interaktiv analys av det politiska språket i Sverige.")

    # News-box CSS
    st.markdown("""
        <style>
        .news-box {
            border: 1px solid #555;           /* En tunn grå ram */
            border-radius: 10px;              /* Mjukt rundade hörn */
            padding: 15px;                    /* Lite luft inuti rutan */
            background-color: transparent;    /* Transparent bakgrund, eller välj en färg t.ex. #1E1E2A */
            margin-bottom: 20px;              /* Lite utrymme under rutan */
        }
        .news-box h3 {
            margin-top: 0;                    /* Tar bort extra utrymme ovanför rubriken */
            margin-bottom: 10px;
            font-size: 1.25em;                /* En lagom stor rubrik */
        }
        .news-box ul {
            list-style-type: none;            /* Tar bort prickarna i listan */
            padding-left: 0;                  /* Tar bort indraget */
            margin-bottom: 0;
        }
        .news-box li {
            margin-bottom: 8px;               /* Lite avstånd mellan varje nyhetsrad */
            font-size: 0.9em;                 /* Något mindre text för nyheterna */
        }
        </style>
    """, unsafe_allow_html=True)

    # # === NY LAYOUT MED TVÅ KOLUMNER ===
    main_col, news_col = st.columns([2, 1])  # Vänster kolumn är dubbelt så bred som den högra
    with main_col:
        # Dashboarddelen
        st.divider()
        live_results_df, live_accuracy, total_live_articles = run_live_evaluation(articles_per_party=4)
    
        total_speeches = fetch_speeches_count()
        latest_speech_date = fetch_latest_speech_date_cached()

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Träffsäkerhet (de {total_live_articles} senaste artiklarna)", f"{live_accuracy:.1f}%")
        col2.metric("Totalt anföranden i databasen", f"{total_speeches:,}".replace(",", " "))
        col3.metric("Senaste anförande", latest_speech_date)
    
        st.divider()
        
        # Info-sektionen
        st.subheader("Mål")
        st.markdown(
            """
            **MaktspråkAI** är ett fullskaligt **data science- och NLP-projekt**.
            Syftet är att **utforska, analysera och visualisera det politiska språkbruket i Sveriges riksdag** genom att kombinera modern maskininlärning och AI med systemdesign.  
            
            Projektet besvarar frågor som:
            - Kan man **förutsäga ett partis tillhörighet** enbart genom språkbruk?
            - Vilka **retoriska mönster** skiljer partierna åt?
            - Hur förändras språket över tid i **debatter**?
            """
        )

        st.divider()

        st.subheader("Teknologi")
        st.markdown(
            """
            Detta projekt är byggt i Python och använder ett antal bibliotek och ramverk för att täcka hela kedjan 
            från datainsamling till analys och visualisering:

            - **Streamlit** Används för att bygga den interaktiva webbapplikationen. Gör det möjligt att testa modeller, 
              visa resultat och utforska data direkt i webbläsaren utan extra konfiguration.  

            - **Pandas & NumPy** Huvudverktyg för datamanipulering och analys. Används för att rensa text, strukturera dataset, 
              hantera tidsserier samt utföra beräkningar och transformationer på miljontals ord och meningar.  

            - **Transformers (Hugging Face)** Kärnan i NLP-delen. Projektet använder och finjusterar modellen **KB/bert-base-swedish-cased** för textklassificering på svenska. Hugging Face-biblioteket möjliggör också enkel laddning av 
              förtränade modeller och jämförelser med alternativa arkitekturer.  

            - **Scikit-learn** Används för att bygga baseline-modeller, utföra evalueringar (precision, recall, F1-score) 
              samt för klassisk textanalys (t.ex. TF-IDF, logistisk regression och SVM).  
              Ger en stabil grund för att jämföra traditionella metoder mot transformer-baserade modeller.  

            - **Plotly, Matplotlib & Calplot** Visualiseringsstacken. Plotly används för interaktiva grafer i webben, Matplotlib för mer 
              klassiska figurer, och Calplot för att skapa kalenderdiagram som visar aktivitetsmönster över tid.  

            - **SQLite** Projektets databas. Hanterar över 30 000 riksdagsanföranden med metadata (parti, datum, talare).  
              ETL-pipelinen laddar automatiskt in nya data, rensar, transformerar och lagrar i SQLite för snabb åtkomst.  

            - **Övrigt**

              Textförbehandling med regex, tokenisering och stopword-listor, klassvikter, weighted sampling, 
              checkpointing och loggning säkerställer reproducerbarhet och att modellen kan tränas och uppdateras smidigt.
            """
        )

        st.divider()

        st.subheader("Om mig")
        st.markdown(
            """
            Jag heter **Martin Blomqvist**.  
            Jag har arbetat med att bygga och förbättra system i olika miljöer – från ekologiskt jordbruk till kod och dataanalys.  
            Oavsett område har fokus varit detsamma: att förstå helheten, hitta struktur och skapa lösningar som fungerar i praktiken.  
            **MaktspråkAI** visar hur jag använder dessa erfarenheter i ett tekniskt sammanhang.  

            
            **Kontakt:**
            - **E-post:** [cm.blomqvist@gmail.com](mailto:cm.blomqvist@gmail.com)
            - **LinkedIn:** [Martin Blomqvist](https://www.linkedin.com/in/martin-blomqvist)
            - **GitHub:** [Martin Blomqvist](https://github.com/martinblomqvistdev)
            """
        )
        
        st.divider()

    # === NYHETSRUTAN HAMNAR I "news_col" ===
    with news_col:
        try:
            news_items = fetch_news()
            if not news_items:
                st.warning("Kunde inte hämta nyhetsflödet.")
            else:
                news_html = '<div class="news-box"><h3>Senaste inrikesnyheterna</h3><ul>'
                for item in news_items:
                    news_html += f'<li><a href="{item["link"]}" target="_blank">{item["title"]}</a></li>'
                news_html += '</ul><div style="text-align: right; font-size: 0.8em; margin-top: 10px;">Från <a href="https://www.svt.se/nyheter/inrikes" target="_blank">SVT Nyheter</a></div></div>'
                
                st.markdown(news_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ett fel uppstod vid hämtning av nyheter.")


# =====================
# Sidebar och Navigation
# =====================
with st.sidebar:
    st.title("MaktspråkAI")
    page = option_menu(
        menu_title=None, 
        options=PAGE_OPTIONS,
        icons=["house-fill", "search", "bar-chart-line-fill", "check2-square", "graph-up"],
        menu_icon="cast", 
        default_index=0,
    )
    st.divider()
    st.header("Filter")

def select_quick_date_range():
    options = {"Senaste 7 dagarna": 7, "Senaste 30 dagarna": 30, "Senaste 90 dagarna": 90, "Detta år": "this_year", "Förra året": "last_year"}
    selected_option = st.sidebar.selectbox("Välj tidsperiod", list(options.keys()))
    today = date.today()
    days = options[selected_option]
    if isinstance(days, int): start_date = today - timedelta(days=days); end_date = today
    elif days == "this_year": start_date = date(today.year, 1, 1); end_date = today
    else: start_date = date(today.year - 1, 1, 1); end_date = date(today.year - 1, 12, 31)
    return start_date, end_date

# =====================
# Huvudlogik för sidvisning
# =====================
if page == "Om projektet":
    welcome_page()

elif page == "Partiprediktion":
    st.header("Partiprediktion")
    st.info("""
        **Testa AI-modellen live!** Klistra in valfri text (t.ex. ett citat, pressmeddelande eller uttalande)
        från ett riksdagsparti. Eller experimentera själv med påhittade citat. Modellen analyserar språk, ton och retorik för att förutsäga vilket parti
        som har skrivit texten. Resultatet visar sannolikheten för alla partier.
    """)
    user_text = st.text_area("Skriv eller klistra in ett citat här:", height=150, label_visibility="collapsed")
    if st.button("Prediktera parti"):
        if user_text.strip():
            with st.spinner("Beräknar…"):
                cleaned_text = clean_text(user_text)
                party_probs = predict_party(model, tokenizer, [cleaned_text])

                # --- Säker fallback om modellen returnerar tomt eller ogiltigt ---
                if not party_probs or not isinstance(party_probs[0], dict):
                    st.warning("Modellen returnerade inget resultat. Kontrollera input eller försök igen.")
                    party_probs = [{p: 0 for p in PARTY_ORDER}]  # fallback med 0 för alla partier

                party_prob_dict = party_probs[0]

                # Hitta parti med högst sannolikhet
                party, prob = max(party_prob_dict.items(), key=lambda x: x[1])

            st.success(f"**Predikterat parti:** {party} ({prob*100:.1f}% säkerhet)")

            fig = px.bar(
                x=PARTY_ORDER,
                y=[party_prob_dict.get(p, 0) for p in PARTY_ORDER],
                labels={"x": "Parti", "y": "Sannolikhet"},
                text=[f"{party_prob_dict.get(p, 0)*100:.1f}%" for p in PARTY_ORDER]
            )
            st.plotly_chart(fig, config={"responsive": True})


elif page == "Språkbruk & Retorik":
    st.header("Jämför partiernas retorik")
    start_date, end_date = select_quick_date_range()
    with st.spinner("Hämtar och analyserar data…"):
        df = fetch_speeches_in_period(start_date, end_date)

    if df.empty:
        st.warning("Kunde inte hitta någon data alls i databasen.")
    else:
        df['parti'] = pd.Categorical(df['parti'], categories=PARTY_ORDER, ordered=True)
        # ÄNDRING: Använder den cachade lexikon-sökvägen
        df_ton = apply_ton_lexicon(df, text_col="text", lexicon_path=LEXICON_PATH)
        ton_columns = [col for col in df_ton.columns if col not in ['text','parti','date']]
        retorik_profil = df_ton.groupby('parti', observed=False)[ton_columns].mean().reindex(PARTY_ORDER).dropna()
        retorik_sammansattning = retorik_profil.div(retorik_profil.sum(axis=1), axis=0) * 100
        tab1, tab2 = st.tabs(["Retoriskt fingeravtryck", "Rankning per kategori"])
        with tab1:
            st.subheader("Partiernas retoriska fingeravtryck")
            df_plot = retorik_sammansattning.reset_index().melt(id_vars='parti', var_name='Kategori', value_name='Andel (%)')
            fig_stacked_bar = px.bar(df_plot, x='parti', y='Andel (%)', color='Kategori', title='Sammansättning av retorik per parti', text_auto='.1f', labels={'parti': 'Parti'})
            fig_stacked_bar.update_layout(xaxis={'categoryorder':'array', 'categoryarray': PARTY_ORDER})
            st.plotly_chart(fig_stacked_bar, config={"responsive": True})
        with tab2:
            st.subheader("Rankning per retorisk kategori")
            category_to_rank = st.selectbox("Välj retorisk kategori:", sorted(retorik_profil.columns))
            if category_to_rank:
                source_df = retorik_profil[[category_to_rank]].copy()
                max_value = source_df[category_to_rank].max()
                source_df['Rankning'] = (source_df[category_to_rank] / max_value) * 100 if max_value > 0 else 0
                
                # Steg 1: Sortera från STÖRST till minst (False = az, True = za)
                ranked_df = source_df.sort_values(by='Rankning', ascending=True)
                
                # Steg 2: Skicka med den sorterade ordningen till Plotly
                fig_bar = px.bar(
                    ranked_df, 
                    x='Rankning', 
                    y=ranked_df.index, 
                    orientation='h', 
                    labels={"y": "Parti", "x": "Relativ poäng (Ledaren = 100)"}, 
                    text_auto='.1f', 
                    title=f"Relativ rankning - {category_to_rank}"
                )
                
                fig_bar.update_layout(yaxis={'categoryorder':'array', 'categoryarray': ranked_df.index.tolist()})

                st.plotly_chart(fig, config={"responsive": True})

        st.divider()
        st.subheader("Vanligaste orden per parti")
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        for i, party in enumerate(retorik_profil.index):
            with cols[i % 4]:
                raw_text_blob = " ".join(df[df["parti"]==party]["text"].dropna().tolist())
                cleaned_text_for_cloud = preprocess_for_wordcloud(raw_text_blob)
                if cleaned_text_for_cloud:
                    wc = WordCloud(width=400, height=300, background_color="white").generate(cleaned_text_for_cloud)
                    st.write(f"**{party}**"); fig_wc, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig_wc); plt.close(fig_wc)

elif page == "Evaluering":
    st.header("Automatisk Testbänk: Prediktion på partiernas egna texter")
    
    # === NY, MER FÖRKLARANDE INFO-BOX ===
    st.info("""
    Hämta och evaluera de senaste texterna direkt från riksdagspartiernas hemsidor. 
    **Notera:** Antalet funna artiklar kan vara lägre än det begärda. Detta beror på att vissa partier har inaktiva RSS-flöden 
    eller att deras senaste inlägg är video-klipp som sållas bort av kvalitetsfiltret.
    """)

    num_per_party = st.slider(
        "Antal senaste artiklar att hämta per parti", 1, 5, 2
    )
    
    show_debug = st.checkbox("Visa felsökningslogg")

    if 'fetch_results' not in st.session_state:
        st.session_state.fetch_results = {"articles": [], "log": [], "found_parties": set()}

    if st.button(f"Hämta & Evaluera upp till {num_per_party * 8} partitexter"):
        with st.spinner(f"Hämtar och analyserar texter... Detta kan ta en stund."):
            st.session_state.fetch_results = fetch_party_articles(articles_per_party=num_per_party)
            # Spara vilka partier vi faktiskt hittade artiklar för
            st.session_state.fetch_results["found_parties"] = {a['true_party'] for a in st.session_state.fetch_results.get("articles", [])}
        st.rerun()

    articles_to_analyze = st.session_state.fetch_results.get("articles", [])
    debug_log = st.session_state.fetch_results.get("log", [])
    found_parties = st.session_state.fetch_results.get("found_parties", set())

    # Om vi har klickat på knappen, visa en sammanfattning
    if debug_log: 
        # === NY SAMMANFATTNING ÖVER RESULTATET ===
        st.subheader("Resultat")
        
        missing_parties = set(PARTY_ORDER) - found_parties
        if missing_parties:
            # Gör om set till en snygg sträng, t.ex. "S, C, KD, MP"
            missing_parties_str = ", ".join(sorted(list(missing_parties)))
            st.warning(f"**Kunde inte hitta giltiga artiklar för:** {missing_parties_str}")

        if articles_to_analyze:
            # (Hela din existerande kod för att visa metrics och tabell)
            results = []
            for article in articles_to_analyze:
                cleaned_for_model = clean_text(article['content'])
                party_probs = predict_party(model, tokenizer, [cleaned_for_model])
                predicted_party = max(party_probs[0].items(), key=lambda x: x[1])[0]
                results.append({
                    "Titel": article['title'], "Sant parti": article['true_party'], "Modellens gissning": predicted_party,
                    "Korrekt?": "✅" if article['true_party'] == predicted_party else "❌", "Länk": article['link']
                })
            results_df = pd.DataFrame(results)
            correct_count = (results_df["Korrekt?"] == "✅").sum()
            total_count = len(results_df)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Antal texter analyserade", f"{total_count}")
            col2.metric("Antal korrekta gissningar", f"{correct_count}")
            col3.metric("Träffsäkerhet", f"{accuracy:.1f}%")
            st.divider()
            st.dataframe(results_df, column_config={"Länk": st.column_config.LinkColumn("Länk", display_text="Öppna artikel")})
        else:
            st.error("Inga giltiga artiklar alls kunde hittas från någon av partiernas flöden.")

    if show_debug:
        st.divider()
        st.subheader("Felsökningslogg")
        st.code("\n".join(debug_log), language="text")

elif page == "Historik":
    st.header("Analysera retorikens utveckling över tid")

    # Definiera de tidsperioder vi vill jämföra (kortast till längst för enkelhet)
    today = date.today()
    time_periods = {
        "Senaste 30 dagarna": (today - timedelta(days=30), today),
        "Senaste 90 dagarna": (today - timedelta(days=90), today),
        "Senaste året": (today - timedelta(days=365), today),
        "Senaste 2 åren": (today - timedelta(days=365*2), today),
        "Senaste 5 åren": (today - timedelta(days=365*5), today),
        "Senaste 10 åren": (today - timedelta(days=365 * 10), today) 
    }

    # Hämta alla unika kategorier för filtret
    lex_df_temp = pd.read_csv(LEXICON_PATH)
    ton_columns = lex_df_temp['kategori'].unique().tolist()

    # Låt användaren välja vilken kategori de vill följa över tid
    category_to_track = st.selectbox(
        "Välj retorisk kategori att följa över tid:", 
        sorted(ton_columns),
        key="historic_category_select"
    )

    all_results = []

    with st.spinner(f"Analyserar historisk data för alla partier i kategorin '{category_to_track}'..."):
        # --- DATABEREDNING FÖR LINJEDIAGRAM ---
        for period_name, (start_date, end_date) in time_periods.items():
            
            # Steg 1: Hämta ALL data för perioden
            df_period = fetch_speeches_in_period(start_date, end_date)

            if not df_period.empty:
                # Steg 2: Applicera lexikonet (viktad version)
                df_ton = apply_ton_lexicon(df_period, text_col="text", lexicon_path=LEXICON_PATH)
                
                # Steg 3: Gruppera på parti och hämta medelpoängen för den valda kategorin
                period_profile = df_ton.groupby('parti', observed=False)[category_to_track].mean().reset_index()
                
                # Steg 4: Lägg till tidsperioden
                period_profile['Period'] = period_name
                period_profile['Period_Sort'] = list(time_periods.keys()).index(period_name)
                all_results.append(period_profile)

    if not all_results:
        st.warning(f"Hittade ingen data alls under de analyserade tidsperioderna.")
    else:
        # Kombinera resultaten
        df_plot = pd.concat(all_results).reset_index(drop=True)
        
        st.subheader(f"Utveckling av retoriken: '{category_to_track}'")
        st.info("Varje linje representerar ett parti. Se hur deras retoriska poäng förändras över tidsperioderna.")

        # Ordna tidsperioderna korrekt (äldst till vänster)
        ordered_periods = list(time_periods.keys())
        
        # Visualisering: linjediagram
        fig = px.line(
            df_plot,
            x="Period", 
            y=category_to_track,
            color="parti",
            title=f"Trend: '{category_to_track}' per parti över tid",
            markers=True,
            category_orders={"Period": ordered_periods} 
        )
        fig.update_xaxes(title_text="Tidsperiod", showgrid=True)
        fig.update_yaxes(
            title_text=f"Genomsnittlig poäng ({category_to_track})",
            range=[df_plot[category_to_track].min() * 0.9, df_plot[category_to_track].max() * 1.1]
        )

        st.plotly_chart(fig, config={"responsive": True})


    st.divider()

    # --- NY SEKTION: 8 ordmoln för jämförelse ---
    st.subheader("Jämför partiernas vanligaste ord")
    st.markdown("Välj en tidsperiod nedan för att se de 8 ordmolnen sida vid sida.")
    
    period_options_reversed = list(time_periods.keys())
    period_options_reversed.reverse()

    period_for_cloud = st.selectbox(
        "Välj period för ordmolnen:",
        period_options_reversed,
        index=0,
        key="all_party_period_select"
    )
    
    # Hämta data för den valda perioden
    start, end = time_periods[period_for_cloud] 
    df_all_data = fetch_speeches_in_period(start, end)[['text', 'parti']]

    if df_all_data.empty:
        st.warning(f"Ingen data hittades för ordmoln under '{period_for_cloud}'.")
    else:
        st.markdown(f"**Visar ordmoln baserat på tal under perioden: {period_for_cloud}**")
        
        cols = st.columns(4)
        
        for i, party in enumerate(PARTY_ORDER):
            with cols[i % 4]:
                df_party = df_all_data[df_all_data['parti'] == party]
                
                if df_party.empty:
                    st.write(f"**{party}** (Ingen data)")
                    st.empty()
                    continue
                
                raw_text_blob = " ".join(df_party["text"].dropna().tolist())
                cleaned_text_for_cloud = preprocess_for_wordcloud(raw_text_blob)
                
                if cleaned_text_for_cloud:
                    try:
                        wc = WordCloud(width=400, height=300, background_color="white", collocations=False).generate(cleaned_text_for_cloud)
                        
                        st.write(f"**{party}**")
                        fig_wc, ax = plt.subplots(figsize=(4, 3))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc, bbox_inches='tight', dpi=fig_wc.dpi)
                        plt.close(fig_wc)
                        
                    except Exception as e:
                        st.error(f"Kunde inte generera moln för {party}.")
                else:
                    st.write(f"**{party}** (För lite text)")

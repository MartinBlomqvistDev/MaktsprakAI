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
    fetch_speeches_historical,
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
    weak_filter_keywords = ["video:", "live:", "se talet", "anförande", "frågestund", "turné", "besöker"]
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
    st.title("MaktspråkAI: Den politiska språkkartan")
    st.markdown("Interaktiv AI-analys av partiernas retorik och mönster.")

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
        col1.metric(f"Träffsäkerhet ({total_live_articles} artiklar)", f"{live_accuracy:.1f}%")
        col2.metric("Totalt anföranden i databasen", f"{total_speeches:,}".replace(",", " "))
        col3.metric("Senaste anförande", latest_speech_date)
    
        st.divider()
        
        st.markdown(
            """
            ### Martin Blomqvist – Om mig och projektet

            Jag heter **Martin Blomqvist** och drivs av att förstå och förbättra komplexa system. Min bakgrund är bred – jag har arbetat i vitt skilda miljöer, från **ekologiskt jordbruk** till avancerad **dataanalys**. Oavsett sammanhang har fokus alltid legat på detsamma: att **hitta den dolda strukturen** i kaoset och bygga lösningar som fungerar i den verkliga världen.
            
            ---
            
            **MaktspråkAI** är en direkt tillämpning av dessa erfarenheter. Det är ett fullskaligt **data science- och NLP-projekt** som skapades under EC Utbildnings Data Scientist-program. Det visar hur jag kombinerar min systemanalytiska förmåga med teknisk kompetens.

            **Projektets mål** är att **utforska, analysera och visualisera det politiska språkbruket i Sveriges riksdag** genom att kombinera modern maskininlärning och AI med robust systemdesign. Jag tar nu steget ut i yrkeslivet via min LIA och ser fram emot att fortsätta utveckla dessa kunskaper och skapa fler användbara produkter. **Följ gärna min fortsatta resa in i detta spännande fält på [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)!**

            ---

            *Nyckelfrågor projektet besvarar:*
            * Kan jag **förutsäga ett partis tillhörighet** enbart genom språkbruk?
            * Vilka **retoriska mönster** skiljer partierna åt i olika frågor?
            * Hur förändras språket över tid i **politiska debatter**?
            """
        )

        st.divider()

        st.markdown(
            """
            ### Teknisk arkitektur: en kraftfull AI-stack

            Detta projekt är byggt på en robust och modern **Python-stack**, utformad för att hantera hela AI-livscykeln – från datainsamling till avancerad NLP och interaktiv visualisering. Jag har valt branschledande verktyg för att säkerställa **skalbarhet, reproducerbarhet** och högsta analysprecision.
            
            ---

            ### Databehandling & modellkärna (the AI engine)

            | Verktyg | Funktion & analysdjup |
            | :--- | :--- |
            | **Transformers (Hugging Face)** | **Kärnan i min NLP-lösning.** Jag utnyttjar och finjusterar **state-of-the-art BERT-modellen (KB/bert-base-swedish-cased)** för banbrytande textklassificering på svenska. Detta möjliggör djup semantisk förståelse och överträffar traditionella metoder i komplexiteten hos politisk text. |
            | **Scikit-learn** | **Modellutvärdering & baslinjeanalys.** Används för att etablera en pålitlig baslinje med klassiska metoder (t.ex. TF-IDF, SVM) och rigorösa evalueringar (**precision, recall, F1-score**). Säkerställer att transformer-modellerna bevisligen förbättrar modellen, även i svåra fall såsom vid snedvriden data. |
            | **Pandas & NumPy** | **Ryggraden i Data Science.** Dessa Python-bibliotek används för effektiv datastrukturering, tidsserieanalys och rensning av miljontals textenheter. Hanterar komplexa beräkningar och transformationer nödvändiga för att förbereda NLP-dataset. |

            ---

            ### Webbapplikation & visualisering (the interface)

            | Verktyg | Funktion & interaktion |
            | :--- | :--- |
            | **Streamlit** | **Interaktiv webbapplikation.** Bygger den snabba och användarvänliga GUI:n. Gör det möjligt för slutanvändare att **omedelbart testa AI-modeller live**, filtrera analysresultat och utforska data direkt i webbläsaren utan någon lokal installation. |
            | **Plotly, Matplotlib & Calplot** | **Dynamisk visualisering.** Ger liv åt datan. **Plotly** skapar interaktiva grafer i applikationen, Matplotlib används för statiska analyser, och Calplot visualiserar aktivitetsmönster och trender över tid. |

            ---

            ### Datainfrastruktur & MLOps

            | Verktyg | Funktion & driftsäkerhet |
            | :--- | :--- |
            | **PostgreSQL (via Supabase)** | **Skalbar databaslösning.** Databasen hanterar effektivt över **40 000 riksdagsanföranden** med komplett metadata. Den driftade PostgreSQL-instansen via Supabase säkerställer **snabb och pålitlig åtkomst** till stora datavolymer. |
            | **ETL & Reproducerbarhet** | **Robust data pipeline.** ETL-pipelinen (Extract, Transform, Load) uppdaterar databasen direkt. Jag använder checkpointing, loggning och weighted sampling för att säkerställa att modellträning är **reproducerbar** och att nya data automatiskt införlivas i analysen. |
            
            ---
            
            ### Kontakt & Portfolio

            * **E-post:** [cm.blomqvist@gmail.com](mailto:cm.blomqvist@gmail.com)
            * **LinkedIn:** [Martin Blomqvist](https://www.linkedin.com/in/martin-blomqvist)
            * **GitHub:** [Martin Blomqvist](https://github.com/martinblomqvistdev)
            """
        )

    # === NYHETSRUTAN MED PARTINAMN SOM UNDERRUBRIK ===
    with news_col:
        try:
            all_articles = fetch_party_articles(articles_per_party=2)["articles"]
            if not all_articles:
                st.warning("Kunde inte hämta partinyheter.")
            else:
                news_html = '<div class="news-box"><h3>Senaste partinyheterna</h3>'
                
                # Gruppera artiklar per parti
                articles_by_party = {}
                for art in all_articles:
                    articles_by_party.setdefault(art["true_party"], []).append(art)
                
                for party, arts in articles_by_party.items():
                    news_html += f'<h4>{party}</h4><ul>'
                    for art in arts:
                        news_html += f'<li><a href="{art["link"]}" target="_blank">{art["title"]}</a></li>'
                    news_html += '</ul>'
                
                news_html += '</div>'
                st.markdown(news_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ett fel uppstod vid hämtning av partinyheter: {e}")



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
        key="sidebar_main_menu"
    )
    st.divider()

    # --- CSS för nyhetsruta i sidebar utan scroll ---
    st.markdown("""
        <style>
        .news-box-sidebar {
            border: 1px solid #555;
            border-radius: 8px;
            padding: 10px;
            background-color: transparent;
            margin-bottom: 15px;
        }

        .news-box-sidebar h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 0.95em;
        }

        .news-box-sidebar ul {
            list-style-type: none;
            padding-left: 0;
            margin-bottom: 0;
        }

        .news-box-sidebar li {
            margin-bottom: 6px;
            font-size: 0.85em;
        }

        .news-box-sidebar a {
            text-decoration: none;
            color: #1E90FF;
        }

        .news-box-sidebar a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Inrikesnyheter i sidebar ---
    try:
        news_items = fetch_news()
        if news_items:
            news_html = '<div class="news-box-sidebar">'
            news_html += '<h3>Senaste inrikesnyheterna</h3><ul>'
            for item in news_items:
                news_html += f'<li><a href="{item["link"]}" target="_blank">{item["title"]}</a></li>'
            news_html += '</ul>'
            news_html += '<div style="text-align: right; font-size: 0.75em; margin-top: 5px;">Från <a href="https://www.svt.se/nyheter/inrikes" target="_blank">SVT Nyheter</a></div>'
            news_html += '</div>'
            st.markdown(news_html, unsafe_allow_html=True)
        else:
            st.info("Inga nyheter kunde hämtas just nu.")
    except Exception:
        st.error("Fel vid hämtning av nyheter.")


# =====================
# Huvudlogik för sidvisning
# =====================
if page == "Om projektet":
    welcome_page()

elif page == "Partiprediktion":
    st.header("Partiprediktion")
    
    # --- Introduktion ---
    st.info("""
    **Testa AI-modellen live!** Klistra in valfri text (t.ex. ett citat, pressmeddelande eller uttalande)
    från ett riksdagsparti. Eller experimentera själv med påhittade citat. Modellen analyserar språk, ton och retorik för att förutsäga vilket parti
    som har skrivit texten.
    """)

    # --- Textarea och knapp ---
    user_text = st.text_area("Skriv eller klistra in ett citat här:", height=150, label_visibility="collapsed")
    prediktion_placeholder = st.empty()  # Vi fyller den med resultatet senare

    if st.button("Prediktera parti"):
        if user_text.strip():
            with st.spinner("Beräknar…"):
                cleaned_text = clean_text(user_text)
                party_probs = predict_party(model, tokenizer, [cleaned_text])

                if not party_probs or not isinstance(party_probs[0], dict):
                    st.warning("Modellen returnerade inget resultat. Kontrollera input eller försök igen.")
                    party_probs = [{p: 0 for p in PARTY_ORDER}]

                party_prob_dict = party_probs[0]
                party, prob = max(party_prob_dict.items(), key=lambda x: x[1])

            # --- Visa resultatet
            with prediktion_placeholder.container():
                st.success(f"**Predikterat parti:** {party} ({prob*100:.1f}% säkerhet)")
                fig = px.bar(
                    x=PARTY_ORDER,
                    y=[party_prob_dict.get(p, 0) for p in PARTY_ORDER],
                    labels={"x": "Parti", "y": "Sannolikhet"},
                    text=[f"{party_prob_dict.get(p, 0)*100:.1f}%" for p in PARTY_ORDER]
                )
                st.plotly_chart(fig, config={"responsive": True})

    # --- Diskret tips/guide, centrerad och smal ---
    cols = st.columns([1, 3, 1])  # 1:3:1 → mittkolumnen blir smalare
    with cols[1]:
        st.info("""
        🧠 **Tips & exempel för att testa modellen**
        
        Här är några autentiska debattcitat du kan prova modellen på:
        
        - "Vi behöver stärka skolan och säkerställa att alla barn får samma möjligheter."  
        Källa: [Aftonbladet Debatt](https://www.aftonbladet.se/debatt)
        - "Miljön är vår tids största utmaning – vi måste agera nu!"  
        Källa: [DN Debatt](https://www.dn.se/debatt/)
        - "Sänk skatterna för att främja företagande och innovation."  
        Källa: [Regeringen Debattartiklar](https://www.regeringen.se/debattartiklar/)
        
        💡 Tips:  
        - Testa påhittade citat eller uttalanden från offentliga personer.  
        - Använd citat från nyhetsartiklar eller offentliga dokument.  
        - Utforska hur modellen tolkar olika retoriska stilar och ämnen.
        """)




elif page == "Språkbruk & Retorik":
    st.header("Jämför partiernas retorik")
    import datetime

    today = datetime.date.today()

    # --- Snabbval för perioder ---
    period_options = {
        "Senaste 1 månad": 30,
        "Senaste 3 månader": 90,
        "Senaste 6 månader": 180,
        "Senaste 12 månader": 365
    }
    selected_period_label = st.selectbox("Välj tidsperiod:", list(period_options.keys()), index=1)  # default 3 månader
    days_delta = period_options[selected_period_label]
    start_date = today - datetime.timedelta(days=days_delta)
    end_date = today
    st.info(f"Visar tal från {start_date} till {end_date}")

    # --- Hämta data ---
    with st.spinner("Hämtar och analyserar data…"):
        df = fetch_speeches_historical(start_date, end_date)

    if df.empty:
        st.warning("Kunde inte hitta någon data alls i databasen.")
    else:
        # --- Ordna partier ---
        df['parti'] = pd.Categorical(df['parti'], categories=PARTY_ORDER, ordered=True)

        # --- Lexikonbaserad tonanalys ---
        df_ton = apply_ton_lexicon(df, text_col="text", lexicon_path=LEXICON_PATH)
        ton_columns = [col for col in df_ton.columns if col not in ['text','parti','protokoll_datum']]

        retorik_profil = df_ton.groupby('parti', observed=False)[ton_columns].mean().reindex(PARTY_ORDER).fillna(0)
        retorik_sammansattning = retorik_profil.div(retorik_profil.sum(axis=1).replace(0,1), axis=0) * 100

        # --- Tabs ---
        tab1, tab2 = st.tabs(["Retoriskt fingeravtryck", "Rankning per kategori"])

        # --- Fingeravtryck ---
        with tab1:
            st.subheader("Partiernas retoriska fingeravtryck")
            df_plot = retorik_sammansattning.reset_index().melt(
                id_vars='parti', var_name='Kategori', value_name='Andel (%)'
            )
            df_plot['Har_data'] = df_plot['Andel (%)'] > 0

            fig_stacked_bar = px.bar(
                df_plot,
                x='parti',
                y='Andel (%)',
                color='Kategori',
                title='Sammansättning av retorik per parti',
                text_auto='.1f',
                labels={'parti': 'Parti'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            # Markera partier utan data
            for parti in PARTY_ORDER:
                if not df_plot[df_plot['parti']==parti]['Har_data'].any():
                    fig_stacked_bar.add_scatter(
                        x=[parti],
                        y=[0],
                        mode='markers',
                        marker=dict(color='lightgrey', size=20),
                        showlegend=False,
                        name=f"{parti} (ingen data)"
                    )

            fig_stacked_bar.update_layout(xaxis={'categoryorder':'array', 'categoryarray': PARTY_ORDER})
            st.plotly_chart(fig_stacked_bar, config={"responsive": True})

        # --- Rankning per kategori ---
        with tab2:
            st.subheader("Rankning per retorisk kategori")
            category_to_rank = st.selectbox("Välj retorisk kategori:", sorted(retorik_profil.columns))
            if category_to_rank:
                source_df = retorik_profil[[category_to_rank]].copy()
                max_value = source_df[category_to_rank].max()
                source_df['Rankning'] = (source_df[category_to_rank] / max_value * 100) if max_value > 0 else 0
                ranked_df = source_df.sort_values(by='Rankning', ascending=True)
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
                st.plotly_chart(fig_bar, config={"responsive": True})

        # --- WordClouds ---
        st.divider()
        st.subheader("Vanligaste orden per parti")
        cols = st.columns(4)
        for i, party in enumerate(PARTY_ORDER):
            with cols[i % 4]:
                raw_text_blob = " ".join(df[df["parti"]==party]["text"].dropna().tolist())
                cleaned_text_for_cloud = preprocess_for_wordcloud(raw_text_blob)
                if cleaned_text_for_cloud:
                    try:
                        wc = WordCloud(width=400, height=300, background_color="white", collocations=False).generate(cleaned_text_for_cloud)
                        st.write(f"**{party}**")
                        fig_wc, ax = plt.subplots(figsize=(4,3))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                        plt.close(fig_wc)
                    except Exception:
                        st.error(f"Kunde inte generera moln för {party}.")
                else:
                    st.write(f"**{party}** (För lite text)")


elif page == "Evaluering":
    st.header("Automatisk Testbänk: Prediktion på partiernas egna texter")
    
    # === NY, MER FÖRKLARANDE INFO-BOX ===
    st.info("""
    Hämta och evaluera de senaste texterna direkt från riksdagspartiernas hemsidor. 
    **Notera:** Antalet funna artiklar kan vara lägre än det begärda då vissa partier har inaktiva RSS-flöden 
    eller att deras senaste inlägg är videoklipp som sållats bort av kvalitetsfiltret.
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

    # --- 1. Definiera tidsgräns och ladda data ---
    MAX_YEARS = 10 
    today = date.today()
    START_DATE_LIMIT = today - timedelta(days=365 * MAX_YEARS) 

    # Läs lexikon och hämta kategorier
    lex_df_temp = pd.read_csv(LEXICON_PATH)
    ton_columns = lex_df_temp['kategori'].unique().tolist()

    # Användarval av kategori
    category_to_track = st.selectbox(
        "Välj retorisk kategori att följa över tid:", 
        sorted(ton_columns),
        key="historic_category_select"
    )

    with st.spinner(f"Analyserar historisk data för alla partier i kategorin '{category_to_track}' tio år tillbaka..."):
        
        # Hämta ALL data inom den maximala tidsperioden
        df_all_data = fetch_speeches_historical("2015-01-01", today)
        
        if df_all_data.empty:
            st.warning(f"Hittade ingen data alls inom den valda tidsgränsen ({START_DATE_LIMIT.year} till {today.year}).")
            st.stop()

        # 2. Robust hantering av DATUM
        df_all_data['protokoll_datum'] = pd.to_datetime(df_all_data['protokoll_datum'], errors='coerce') 
        valid_dates_df = df_all_data.dropna(subset=['protokoll_datum'])

        # Debug-info: visa både efterfrågat och faktiskt intervall
        requested_range = f"{START_DATE_LIMIT} → {today}"
        if not valid_dates_df.empty:
            min_date = valid_dates_df['protokoll_datum'].min().strftime('%Y-%m-%d')
            max_date = valid_dates_df['protokoll_datum'].max().strftime('%Y-%m-%d')
            st.info(f"Efterfrågad period: **{requested_range}**\n\nData i databasen täcker: **{min_date} → {max_date}**")
        else:
            st.warning("Hittade inga giltiga datum i den hämtade datan efter rensning.")
            st.stop()

        # 3. Aggregera till KURVOR (per ÅR)
        df_ton = apply_ton_lexicon(valid_dates_df, text_col="text", lexicon_path=LEXICON_PATH)
        df_ton['År'] = df_ton['protokoll_datum'].dt.to_period('Y') 
        df_plot_yearly = df_ton.groupby(['parti', 'År'], observed=False)[category_to_track].mean().reset_index()
        df_plot_yearly['År'] = df_plot_yearly['År'].astype(str).str.split('-').str[0].astype(int) 
        unique_years = sorted(df_plot_yearly['År'].unique())

        # Konvertera till procent av partiets tal
        df_plot_yearly[category_to_track] = df_plot_yearly[category_to_track] * 100

        # 4. Visualisering
        st.subheader(f"Utveckling av retoriken: '{category_to_track}'")
        st.markdown(f"Visar trenden för de senaste {MAX_YEARS} åren med årlig upplösning.")

        fig = px.line(
            df_plot_yearly,
            x="År",
            y=category_to_track,
            color="parti",
            markers=True,
            title=f"Trend: '{category_to_track}' per parti över tid (Årlig upplösning)"
        )

        fig.update_xaxes(
            title_text="År", 
            tickvals=unique_years, 
            ticktext=[str(year) for year in unique_years], 
            showgrid=True
        )
        fig.update_yaxes(
            title_text=f"% av partiets tal med kategori '{category_to_track}'", 
            range=[df_plot_yearly[category_to_track].min() * 0.9,
                   df_plot_yearly[category_to_track].max() * 1.1]
        )
        fig.update_traces(hovertemplate='%{y:.1f}% av partiets tal')

        st.plotly_chart(fig, config={"responsive": True})

    st.divider()

    # --- WordClouds per parti ---
    st.subheader("Jämför partiernas vanligaste ord")
    st.markdown("Välj en tidsperiod nedan för att se ordmoln sida vid sida.")

    time_periods_for_cloud = {
        "Senaste 10 åren": (today - timedelta(days=365*10), today),
        "Senaste 5 åren": (today - timedelta(days=365*5), today),
        "Senaste 2 åren": (today - timedelta(days=365*2), today),
        "Senaste året": (today - timedelta(days=365), today),
        "Senaste 90 dagarna": (today - timedelta(days=90), today),
        "Senaste 30 dagarna": (today - timedelta(days=30), today)
    }
    
    period_options_reversed = list(time_periods_for_cloud.keys())[::-1]
    period_for_cloud = st.selectbox(
        "Välj period för ordmolnen:",
        period_options_reversed,
        index=0,
        key="all_party_period_select"
    )

    start, end = time_periods_for_cloud[period_for_cloud]
    df_all_data_cloud = fetch_speeches_historical(start, end)[['text', 'parti']]

    if df_all_data_cloud.empty:
        st.warning(f"Ingen data hittades för ordmoln under '{period_for_cloud}'.")
    else:
        st.markdown(f"**Visar ordmoln baserat på tal under perioden: {period_for_cloud}**")
        cols = st.columns(4)

        for i, party in enumerate(PARTY_ORDER):
            with cols[i % 4]:
                df_party = df_all_data_cloud[df_all_data_cloud['parti'] == party]
                
                if df_party.empty:
                    st.write(f"**{party}** (Ingen data)")
                    continue

                raw_text_blob = " ".join(df_party["text"].dropna().tolist())
                cleaned_text_for_cloud = preprocess_for_wordcloud(raw_text_blob)

                if cleaned_text_for_cloud:
                    try:
                        wc = WordCloud(
                            width=400,
                            height=300,
                            background_color="white",
                            collocations=False
                        ).generate(cleaned_text_for_cloud)
                        st.write(f"**{party}**")
                        fig_wc, ax = plt.subplots(figsize=(4, 3))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc, bbox_inches='tight', dpi=fig_wc.dpi)
                        plt.close(fig_wc)
                    except Exception:
                        st.error(f"Kunde inte generera moln för {party}.")
                else:
                    st.write(f"**{party}** (För lite text)")

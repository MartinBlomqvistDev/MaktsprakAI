# src/maktsprak_pipeline/nlp.py (SLUTLIG REVIDERING MED MFL OCH RES)
__all__ = ["apply_ton_lexicon", "combined_stopwords", "clean_text"]

from pathlib import Path
import pandas as pd
import re 

# ===================== Projektroot och stopwords =====================
proj_root = Path(__file__).parents[2]
stopwords_path = proj_root / "data/processed/stopwords-sv.txt"
with open(stopwords_path, "r", encoding="utf-8") as f:
    swedish_stopwords = {line.strip().lower() for line in f if line.strip()}

political_stopwords = {
    # Titlar och roller
    "herr", "fru", "talman", "statsrådet", "ministern", "finansministern", "ledamot", "ledamoten",
    "partiledare", "tjänstgörande", "ersättare", "vice", "andre", "tredje", "förste",
    
    # Debatt-specifika termer (MFL och RES TILLAGDA HÄR)
    "anförande", "replik", "interpellation", "interpellationer", "fråga", "frågan", "frågor",
    "svar", "svaret", "debatt", "debatten", "kammaren", "protokoll", "prot", "utskottet",
    "betänkande", "ärende", "yrkande", "bifall", "avslag", "tackar",
    "mfl", "res", # <-- NYA STOPPORD FÖR RIKSDAGSPROTOKOLL
    
    # Allmänna politiska termer 
    "regeringen", "regeringens", "riksdagen", "riksdagens", "partiet", "partierna", "partiernas", 
    "sverige", "landet", "svensk", "svenska", "land", "nation", "medborgare", "medborgarna",
    "samhället", "frågeställning", "dessa", "andra", 
    "politik", "politiken", "förslag", "budgeten", "miljarder", "kronor", "procent", "sveriges"
    
    # Vanliga verb och substantiv i debatter
    "tack", "gäller", "finns", "handlar", "betyder", "innebär", "anser", "tycker",
    "menar", "tror", "göra", "säga", "se", "vet", "behöver", "borde", "självklart",
    "därför", "också", "samtidigt", "väldigt", "helt", "bara", "kanske", "ytterligare",
    "tid", "gång", "nya", "stora", "olika", "viktigt", "får",
    
    # Grundläggande stoppord (extra säkerhet)
    "vi", "att", "för", "på", "och", "men", "eller", "nu", "som", "med", "är", "den", "det",
    "ett", "en", "av", "om", "till", "har", "vår", "vårt", "våra", "de", "dem", "dig", "oss",
    "måste", "skall", "ska", "barn", "både", "människor", "ändå", "bör", "åtgärder", "stöd", "sveriges",
    "uppdrag", "staten", "personer", "person", "talma", "fortsätt", "fortsätta", "mar"
}
combined_stopwords = swedish_stopwords.union(political_stopwords)

# ===================== Text-rensning (för AI-modellen) =====================
def clean_text(text: str) -> str:
    """Rensar text genom att ta bort radbrytningar och extra mellanslag i ändarna."""
    # OBS: Om din clean_text använde regex tidigare, kan du återinföra den här.
    return text.replace("\n", " ").strip() if isinstance(text, str) else ""

# ===================== Applicera tonlexikon (Med VIKTNING) =====================
def apply_ton_lexicon(df, text_col="text", lexicon_path=None):
    if lexicon_path is None or not lexicon_path.exists():
        return df
    
    # Läs in lexikonet och säkerställ kolumnnamnen
    lex_df = pd.read_csv(lexicon_path)
    # Justera för det nya formatet: ord,kategori,vikt
    lex_df.columns = ['ord', 'kategori', 'vikt'] 
    
    categories = lex_df['kategori'].unique().tolist()
    
    # Skapa en ordlista där varje ord mappar till sin vikt
    word_to_weight = lex_df.set_index('ord')['vikt'].to_dict()
    
    # Skapa en dictionary där varje kategori har en lista av sina ord för snabb sökning
    lexicon_sets = {cat: set(lex_df.loc[lex_df['kategori']==cat, 'ord'].values) for cat in categories}

    for cat in categories:
        df[cat] = 0.0
        
    for idx, row in df.iterrows():
        # Dela upp texten i ord
        words = str(row[text_col]).lower().split()
        total_words = len(words)
        
        if total_words == 0:
            continue
            
        for cat in categories:
            # Beräkna total vikt istället för bara antal
            total_weight = sum(word_to_weight.get(w, 0) for w in words if w in lexicon_sets[cat])
            
            # Formeln för poäng: (Total vikt / Antal ord) * 100
            df.at[idx, cat] = (total_weight / total_words) * 100
            
    return df
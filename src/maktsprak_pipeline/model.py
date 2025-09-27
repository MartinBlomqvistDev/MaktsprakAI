# src/maktsprak_pipeline/model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig # <-- LADE TILL AutoConfig
from torch.nn.functional import softmax

# Din Hugging Face-repo-sträng
MODEL_NAME_OR_PATH = "MartinBlomqvist/maktsprak_classifier_clean"

# Definiera partierna i den ordning modellen förväntar sig dem
# Denna ordning måste matcha 'LABELS' från träningen
PARTIES = ["C", "KD", "L", "M", "MP", "S", "SD", "V"]

def load_model_and_tokenizer(device=None):
    """
    Laddar en finjusterad modell och tokenizer direkt från Hugging Face Model Hub.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Laddar modell och tokenizer från Hugging Face sökväg: {MODEL_NAME_OR_PATH}")
    
    # FIX 1: Tvinga laddning av config separat för robusthet
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
    
    # Ladda allt i ett svep. 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    # VIKTIGT: Skicka med config till modellen för att säkerställa korrekt laddning
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, config=config)
    
    model.to(device)
    model.eval()
    print("Modell och tokenizer har laddats färdigt.")
    return model, tokenizer

def predict_party(model, tokenizer, texts):
    # Hämta den aktiva enheten
    device = next(model.parameters()).device
    results = []
    
    # FIX 2: Använder PARTIES-listan direkt för mappning (löser model.config.id2label-felet)
    id2label = {i: party for i, party in enumerate(PARTIES)} 
    
    for text in texts:
        # Skicka texten direkt till tokenizer utan egen förbehandling
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=1).squeeze().cpu().tolist()
        
        # Mappa sannolikheterna till rätt partinamn baserat på vår nya id2label
        results.append({id2label[i]: prob for i, prob in enumerate(probs)})
        
    # FIX 3: Rättar stavfelet från 'return resultsgi' till 'return results'
    return results
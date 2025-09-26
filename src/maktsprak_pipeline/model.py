# src/maktsprak_pipeline/model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

    # Uppdatera print-texten för att visa att den laddar från HF
    print(f"Laddar modell och tokenizer från Hugging Face sökväg: {MODEL_NAME_OR_PATH}")
    
    # Ladda allt i ett svep. Transformers hämtar filerna från MODEL_NAME_OR_PATH.
    # VIKTIGT: Använd AutoTokenizer/AutoModel för att ladda från Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
    
    model.to(device)
    model.eval()
    print("Modell och tokenizer har laddats färdigt.")
    return model, tokenizer

def predict_party(model, tokenizer, texts):
    # Denna funktion behöver inga ändringar
    device = next(model.parameters()).device
    results = []
    
    id2label = model.config.id2label
    
    for text in texts:
        # Skicka texten direkt till tokenizer utan egen förbehandling
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=1).squeeze().cpu().tolist()
        
        # Mappa sannolikheterna till rätt partinamn baserat på modellens egen konfiguration
        results.append({id2label[i]: prob for i, prob in enumerate(probs)})
        
    return results
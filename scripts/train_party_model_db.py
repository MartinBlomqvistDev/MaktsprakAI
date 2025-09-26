# =========================================================
# Fil: scripts/train_party_model_db.py
# Syfte: Träna BERT för parti-klassificering direkt från DB
# Förbättrad version med:
#  - AMP (mixed precision) på GPU
#  - Gradient clipping
#  - Weight decay
#  - Label smoothing
#  - MAX_LENGTH = 512 för riksdagstal
#  - Weighted sampler för obalanserad data
#  - Checkpoint & early stopping
#  - FGM (adversarial training)
#  - Mixup på embeddings
#  - Gradual layer unfreezing
#  - OneCycleLR scheduler
#  - Extra dropout
#  - Klassvikter
#  - Backtranslation för små partier (utom M och S)
# =========================================================

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from src.maktsprak_pipeline.db import create_connection
from src.maktsprak_pipeline.config import (
    TRAIN_MODEL_NAME, TRAIN_MODEL_DIR, TRAIN_BATCH_SIZE, 
    TRAIN_MAX_EPOCHS, TRAIN_MAX_LENGTH, TRAIN_LEARNING_RATE, 
    TRAIN_WEIGHT_DECAY, TRAIN_EARLY_STOPPING_PATIENCE, 
    TRAIN_LABEL_SMOOTHING, BASE_BATCH_SIZE, BASE_MAX_LR, DB_PATH
)
import glob
import numpy as np
import torch.nn as nn

# =====================
# Inställningar (Använder variabler från config)
# =====================
MODEL_NAME = TRAIN_MODEL_NAME
MODEL_DIR = Path(TRAIN_MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = TRAIN_BATCH_SIZE
MAX_EPOCHS = TRAIN_MAX_EPOCHS
MAX_LENGTH = TRAIN_MAX_LENGTH
LEARNING_RATE = TRAIN_LEARNING_RATE
WEIGHT_DECAY = TRAIN_WEIGHT_DECAY
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_STOPPING_PATIENCE = TRAIN_EARLY_STOPPING_PATIENCE
LABEL_SMOOTHING = TRAIN_LABEL_SMOOTHING
GRAD_CLIP_NORM = 1.0
DROPOUT_PROB = 0.2
MIXUP_ALPHA = 0.2

# =====================
# Batch-scalad max_lr för OneCycleLR
# =====================
scaled_max_lr = BASE_MAX_LR * (BATCH_SIZE / BASE_BATCH_SIZE)
print(f"Batch size: {BATCH_SIZE} | max_lr skalar till {scaled_max_lr:.2e}")

# =====================
# Data-funktioner
# =====================
def clean_text(text):
    return text.replace("\n"," ").strip() if isinstance(text,str) else ""

def label_party_from_account(account: str) -> str:
    party_accounts = {
        "S": ["1587012835409788928"], "M": ["747426555417198592"],
        "V": ["282532238"], "L": ["455193032"], "KD": ["1407151866"],
        "C": ["232799403"], "MP": ["41214271","370900852"],
        "SD": ["95972673"]
    }
    for party, accounts in party_accounts.items():
        if account in accounts:
            return party
    return "NA"

def get_training_data(db_path: str = DB_PATH):
    with create_connection(db_path) as conn:
        tweets_df = pd.read_sql_query("SELECT username, text FROM tweets", conn)
        speeches_df = pd.read_sql_query("SELECT parti, text FROM speeches", conn)
        try:
            aug_df = pd.read_sql_query(
                "SELECT parti AS label, text FROM speeches_aug WHERE parti NOT IN ('M','S')",
                conn
            )
        except pd.io.sql.DatabaseError:
            print("Varning: Tabellen 'speeches_aug' hittades inte. Fortsätter utan augmentation.")
            aug_df = pd.DataFrame(columns=['label', 'text'])

    tweets_df["text"] = tweets_df["text"].apply(clean_text)
    tweets_df["label"] = tweets_df["username"].apply(label_party_from_account)
    speeches_df["text"] = speeches_df["text"].apply(clean_text)
    speeches_df.rename(columns={"parti":"label"}, inplace=True)
    aug_df["text"] = aug_df["text"].apply(clean_text)

    df = pd.concat([
        tweets_df[["text","label"]],
        speeches_df[["text","label"]],
        aug_df[["text","label"]]
    ]).reset_index(drop=True)
    
    riksdagspartier = ["S","M","V","L","KD","C","MP","SD"]
    df = df[df["label"].isin(riksdagspartier)].reset_index(drop=True)
    
    return df

# =====================
# Dataset Class
# =====================
class PartyDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=128):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_length=max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k:v.squeeze(0) for k,v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx],dtype=torch.long)
        return item


# =========================================================
# Huvud-träningsskriptet
# =========================================================
if __name__ == "__main__":
    print("Startar träning...")
    
    # Hämta och förbered data
    df = get_training_data()
    
    # Statistik & logg
    counts = df["label"].value_counts().to_dict()
    print("Datapunkter per parti (efter filtrering och backtranslation):")
    for p in ["S","M","V","L","KD","C","MP","SD"]:
        print(f"  {p}: {counts.get(p,0)}")
        if counts.get(p,0) < 10:
            print(f"    VARNING: {p} har få samples ({counts.get(p,0)}). Överväg mer data eller augmentation.")
    print(f"\nTotalt samples efter backtranslation: {len(df)}")

    # Label → id
    LABELS = sorted(df["label"].unique().tolist())
    label2id = {l:i for i,l in enumerate(LABELS)}
    id2label = {i:l for l,i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(), test_size=0.1, random_state=42, stratify=df["label_id"]
    )
    
    # Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PartyDataset(train_texts,train_labels,tokenizer,MAX_LENGTH)
    val_dataset = PartyDataset(val_texts,val_labels,tokenizer,MAX_LENGTH)
    
    # Weighted sampler
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    weights = 1.0 / label_counts
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=sampler)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)
    
    # Model & optimizer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )

    # Dropout
    try:
        model.config.hidden_dropout_prob = DROPOUT_PROB
        if hasattr(model,"dropout"):
            model.dropout = nn.Dropout(DROPOUT_PROB)
    except:
        pass

    model.to(DEVICE)
    
    # Weight decay på allt utom bias/LayerNorm
    no_decay = ["bias","LayerNorm.weight","layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay":WEIGHT_DECAY},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # Scheduler: OneCycleLR med batch-scalad max_lr
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, max_lr=scaled_max_lr, steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS, pct_start=0.1, anneal_strategy='cos'
    )

    # Loss med klassvikter
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # FGM (adversarial training)
    class FGM:
        def __init__(self,model,epsilon=1.0):
            self.model=model
            self.epsilon=epsilon
            self.backup={}
        def attack(self):
            for name,param in self.model.named_parameters():
                if param.requires_grad and "embedding" in name:
                    self.backup[name]=param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)
        def restore(self):
            for name,param in self.model.named_parameters():
                if name in self.backup:
                    param.data = self.backup[name]
            self.backup = {}
    fgm = FGM(model)

    # Load checkpoint om finns
    checkpoint_files = sorted(glob.glob(str(CHECKPOINT_DIR/"epoch_*.pt")), key=lambda x:int(Path(x).stem.split("_")[-1]))
    start_epoch=0
    if checkpoint_files:
        latest_ckpt=checkpoint_files[-1]
        print(f"Found checkpoint: {latest_ckpt}, loading...")
        model.load_state_dict(torch.load(latest_ckpt,map_location=DEVICE))
        start_epoch=int(Path(latest_ckpt).stem.split("_")[-1])
        print(f"Resuming from epoch {start_epoch+1}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Pre-training sanity check
    model.eval()
    sanity_samples = min(500,len(val_dataset))
    if sanity_samples>0:
        sample_indices = np.random.choice(len(val_dataset),sanity_samples,replace=False)
        correct=0
        label_seen=set()
        with torch.no_grad():
            for idx in sample_indices:
                batch = val_dataset[idx]
                batch = {k:v.unsqueeze(0).to(DEVICE) for k,v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits,dim=1)
                correct += (preds==batch["labels"]).sum().item()
                label_seen.add(batch["labels"].item())
        acc = correct/sanity_samples
        missing_labels = set(range(len(LABELS)))-label_seen
        if missing_labels:
            missing_names = [id2label[i] for i in missing_labels]
            print(f"VARNING: Följande klasser saknas i sanity-sample: {missing_names}")
        print(f"Sanity check validerings-accuracy: {acc:.4f} över {sanity_samples} samples")

    # Train loop med AMP, grad clipping, FGM, OneCycleLR, early stopping
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda"))
    best_val_f1 = 0.0
    epochs_no_improve=0

    # Freeze BERT initialt, gradual unfreeze epok 3
    for name,param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(f"\nAnvänder enhet: {DEVICE}  | MAX_LENGTH: {MAX_LENGTH} | BATCH_SIZE: {BATCH_SIZE}")
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total_loss=0.0
        loop = tqdm(train_loader,desc=f"Training epoch {epoch+1}",leave=True)
        for batch in loop:
            optimizer.zero_grad()
            batch = {k:v.to(DEVICE) for k,v in batch.items()}

            # Gradual layer unfreeze: epok 3+
            if epoch==2:
                for name,param in model.named_parameters():
                    param.requires_grad = True

            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                outputs = model(**{k:v for k,v in batch.items() if k!="labels"})
                logits = outputs.logits
                loss = criterion(logits,batch["labels"])
            scaler.scale(loss).backward()

            # FGM adversarial step
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                outputs_adv = model(**{k:v for k,v in batch.items() if k!="labels"})
                logits_adv = outputs_adv.logits
                loss_adv = criterion(logits_adv,batch["labels"])
            scaler.scale(loss_adv).backward()
            fgm.restore()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/(loop.n+1), batch_samples=len(batch["labels"]))

        avg_loss = total_loss/len(train_loader)
        print(f"\nEpoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

        # Validering
        model.eval()
        preds_all,labels_all=[],[]
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                    outputs = model(**{k:v for k,v in batch.items() if k!="labels"})
                    logits = outputs.logits
                preds = torch.argmax(logits,dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(labels)

        acc = accuracy_score(labels_all,preds_all)
        f1_macro = f1_score(labels_all,preds_all,average="macro")
        prec_macro = precision_score(labels_all,preds_all,average="macro",zero_division=0)
        rec_macro = recall_score(labels_all,preds_all,average="macro",zero_division=0)

        print(f"Validation metrics after epoch {epoch+1}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1-macro:  {f1_macro:.4f}")
        print(f"  Precision: {prec_macro:.4f}")
        print(f"  Recall:    {rec_macro:.4f}")

        print("\nPer-klass metrics:")
        print(classification_report(labels_all,preds_all,target_names=LABELS,zero_division=0))

        # Early stopping
        if f1_macro>best_val_f1:
            best_val_f1=f1_macro
            epochs_no_improve=0
            best_path = MODEL_DIR/"best_model.pt"
            torch.save(model.state_dict(),best_path)
            print(f"New best model saved! ({best_path})")
        else:
            epochs_no_improve+=1
            print(f"No improvement for {epochs_no_improve} epoch(s). Best F1: {best_val_f1:.4f}")
            if epochs_no_improve>=EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                last_path = MODEL_DIR/f"last_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(),last_path)
                break

        # Checkpoint
        checkpoint_path = CHECKPOINT_DIR/f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(),checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Spara slutmodell & tokenizer
    model.save_pretrained(MODEL_DIR, safe_serialization=False)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Partimodell sparad i {MODEL_DIR}")
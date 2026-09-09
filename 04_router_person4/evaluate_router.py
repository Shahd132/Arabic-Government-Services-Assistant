import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "models/router_model"
DATA_DIR = "data"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f"{MODEL_DIR}/label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {int(k): v for k, v in label_map["id2label"].items()}

test_df = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

preds = []
with torch.no_grad():
    for i in range(0, len(test_df), 32):
        batch = test_df["question"].iloc[i:i+32].tolist()
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

y_true = test_df["label"].tolist()
target_names = [id2label[i] for i in range(len(id2label))]

print(classification_report(y_true, preds, target_names=target_names))
print("Confusion matrix:")
print(confusion_matrix(y_true, preds))
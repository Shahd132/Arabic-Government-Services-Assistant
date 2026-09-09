import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/router_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f"{MODEL_DIR}/label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {int(k): v for k, v in label_map["id2label"].items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict(text: str):
    enc = tokenizer(text, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    return id2label[pred_id], probs[pred_id].item()

if __name__ == "__main__":
    while True:
        q = input("\nسؤال المستخدم (exit للخروج): ")
        if q.strip().lower() == "exit":
            break
        label, conf = predict(q)
        print(f"-> Category: {label}  (confidence: {conf:.2%})")
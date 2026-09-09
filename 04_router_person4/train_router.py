import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import evaluate

MODEL_NAME = "UBC-NLP/MARBERT"   # موديل مدرب على لهجات عربية (مناسب للمصري)
DATA_DIR = "data"
OUT_DIR = "models/router_model"

with open(f"{DATA_DIR}/label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = {int(k): v for k, v in label_map["id2label"].items()}

train_df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv", encoding="utf-8-sig")

train_ds = Dataset.from_pandas(train_df[["question", "label"]], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[["question", "label"]], preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["question"], truncation=True, max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

args = TrainingArguments(
    output_dir="models/checkpoints",
    eval_strategy="epoch",          # لو طلع Error جرب evaluation_strategy بدلها
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,     # <-- ده بدل tokenizer=tokenizer القديمة
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
with open(f"{OUT_DIR}/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Saved to", OUT_DIR)
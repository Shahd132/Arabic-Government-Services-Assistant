import pandas as pd
from sklearn.model_selection import train_test_split
import json

DATA_PATH = "data/intent_dataset.csv"
OUT_DIR = "data"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df[["question", "category"]].dropna()

labels = sorted(df["category"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

df["label"] = df["category"].map(label2id)

# 70% train / 15% val / 15% test - stratified عشان التوزيع يفضل متوازن
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

train_df.to_csv(f"{OUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False, encoding="utf-8-sig")

with open(f"{OUT_DIR}/label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print(label2id)
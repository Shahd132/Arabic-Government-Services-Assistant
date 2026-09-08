import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import re
import argparse

BOILERPLATE_PREFIX = (
    "قد تحاول الوصول إلى هذا الموقع من مستعرض آمن موجود على الخادم. "
    "يرجى تمكين البرامج النصية وإعادة تحميل هذه الصفحة. "
    "تفاصيل الخدمة الرئيسية > خدماتنا > خدمات إرشادية ومعلوماتية > تفاصيل الخدمة "
    "نماذج خاصة بالخدمة المستندات و الأوراق المطلوبة الرسوم والمبالغ المطلوبة ملاحظات"
)

ZERO_WIDTH_SPACE = "\u200b"

STEPS_JSON_PATTERN = re.compile(r"\[\s*\{.*?\"الخطوات\".*?\}\s*\]", re.DOTALL)


def clean_zero_width(text: str) -> str:
    return text.replace(ZERO_WIDTH_SPACE, "")


def strip_boilerplate(text: str) -> str:
    if text.startswith(BOILERPLATE_PREFIX):
        text = text[len(BOILERPLATE_PREFIX):]
    return text.strip()


def reformat_steps_json(text: str) -> str:
    match = STEPS_JSON_PATTERN.search(text)
    if not match:
        return text
    raw_block = match.group(0)
    try:
        steps = json.loads(raw_block)
        step_texts = [s.get("الخطوات", "").strip() for s in steps if s.get("الخطوات")]
        numbered = " ".join(f"الخطوة {i+1}: {t}" for i, t in enumerate(step_texts))
        return text[:match.start()] + numbered + text[match.end():]
    except json.JSONDecodeError:
        return text[:match.start()] + text[match.end():]


def clean_text(raw_text: str) -> str:
    text = clean_zero_width(raw_text)
    text = strip_boilerplate(text)
    text = reformat_steps_json(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert(raw_pages, department="traffic", categories=None):
    categories = categories or ["قانون المرور"]
    documents = []
    skipped = []

    for page in raw_pages:
        label = page["label"]
        raw_text = page.get("text", "")
        url = page.get("url", "")

        if not raw_text.strip():
            skipped.append(label)
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            skipped.append(label)
            continue

        documents.append({
            "title": label,
            "department": department,
            "categories": categories,
            "source_url": url,
            "source": "traffic.moi.gov.eg (scraped)",
            "text": cleaned,
        })

    return documents, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="traffic_scraped_raw.json")
    parser.add_argument("--output", default="traffic_scraped_clean.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    documents, skipped = convert(raw_pages)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {len(documents)} documents from {len(raw_pages)} scraped pages")
    if skipped:
        print(f"Skipped {len(skipped)} pages with empty text: {skipped}")
    print(f"Saved to {args.output} -- ready to hand to Person 2 for chunking")


if __name__ == "__main__":
    main()
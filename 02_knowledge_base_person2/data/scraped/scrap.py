import requests
from bs4 import BeautifulSoup
import json
import re
import time

BASE = "https://traffic.moi.gov.eg"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ar,en;q=0.9",
}

# Pages worth scraping, with a label for each
targets = [
    ("/Arabic/Pages/FAQ.aspx", "الأسئلة الشائعة"),
    ("/Arabic/Pages/terms-vehicleslicense-renewal.aspx", "تجديد رخصة المركبة"),
    ("/Arabic/Pages/terms-vehicleslicense-replacement-of-lost.aspx", "بدل فاقد رخصة المركبة"),
    ("/Arabic/Pages/terms-vehicleslicense-replacement-of-damaged.aspx", "بدل تالف رخصة المركبة"),
    ("/Arabic/Pages/terms-drivinglicense-replacement-of-lost.aspx", "بدل فاقد رخصة القيادة"),
    ("/Arabic/Pages/terms-drivinglicense-replacement-of-damaged.aspx", "بدل تالف رخصة القيادة"),
    ("/Arabic/Pages/terms-appointment-booking.aspx", "حجز موعد بوحدات المرور"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=2", "نقل قيد أو ملكية مركبة"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=3", "استخراج رخصة تسيير مركبة لأول مرة"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=4", "ترخيص تجارى"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=5", "تجديد رخصة تسيير مركبة"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=6", "بدل فاقد أو تالف لرخص تسيير المركبات"),
    ("/Arabic/OurServices/InfoServices/Pages/ServiceDetails.aspx?ServiceID=7", "استخراج شهادة بيانات للمركبات"),
]

def extract_main_text(html):
    """Pull readable paragraph text out of the page, skipping nav/menu/footer clutter."""
    soup = BeautifulSoup(html, "lxml")

    # Remove known noise elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # SharePoint pages usually put real content inside a main content placeholder div
    # Try common containers first, fall back to <body> if not found
    content_area = (
        soup.find(id=re.compile("MSO_ContentTable", re.I))
        or soup.find(class_=re.compile("ms-rtestate-field", re.I))
        or soup.find("main")
        or soup.body
    )

    if not content_area:
        return ""

    text = content_area.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


results = []
for path, label in targets:
    url = BASE + path
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        print(f"{resp.status_code}  {label}  ({url})")
        if resp.status_code == 200:
            text = extract_main_text(resp.text)
            results.append({"label": label, "url": url, "text": text})
        time.sleep(1)  # be polite to the server
    except Exception as e:
        print(f"ERROR on {url}: {e}")

# Save raw scraped text for inspection before chunking
with open("traffic_scraped_raw.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(results)} pages to traffic_scraped_raw.json")
for r in results:
    print(f"\n--- {r['label']} ({len(r['text'])} chars) ---")
    print(r['text'][:200], "...")
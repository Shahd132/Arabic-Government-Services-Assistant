import sys, os, io

# Force UTF-8 output so Arabic text doesn't break on Windows (cp1252 default)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from retrieve import retrieve
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from retrieve import retrieve
demo_queries = [
    ("ازاي اجدد رخصة القيادة؟", "traffic"),
    ("عقوبة القيادة بدون رخصة", "traffic"),
    ("شروط سن الزواج القانونية", "civil_affairs"),      # ← replaces the ID card query
    ("حقوق الزوجة في النفقة", "civil_affairs"),          # ← another one clearly covered
    ("نسبة ضريبة القيمة المضافة", "tax"),
]

print("=" * 90)
print("Smart Citizen Assistant — Hybrid Retrieval Demo (Person 3)")
print("=" * 90)

for query, dept in demo_queries:
    print(f"\nQuery: {query}")
    print(f"Department: {dept}")
    print("-" * 90)
    results = retrieve(query, top_k=3, department=dept)
    for i, r in enumerate(results, start=1):
        dense_str = f"{r.dense_score:.4f}" if r.dense_score is not None else "N/A"
        sparse_str = f"{r.sparse_score:.4f}" if r.sparse_score is not None else "N/A"
        print(f"  #{i} | score={r.score:.4f} | dense={dense_str} | sparse={sparse_str}")
        print(f"      Law: {r.chunk.law_name}")
        print(f"      Text: {r.chunk.text[:120]}...")
    print()

print("=" * 90)
print("Demo complete.")
print("=" * 90)

# import json, glob

# files = glob.glob(r'..\02_knowledge_base_person2\output\chunks\traffic\*.json')
# terms = ['تجديد', 'تجديد رخصه', 'تجديد الرخصه', 'تجديد رخصة']

# all_chunks = []
# for f in files:
#     data = json.load(open(f, encoding='utf-8'))
#     if isinstance(data, dict):
#         data = [data]
#     all_chunks.extend(data)

# print('Total traffic chunks:', len(all_chunks))
# for t in terms:
#     matches = [d for d in all_chunks if t in d.get('text', '')]
#     print(f"TERM: {t} MATCHES: {len(matches)}")
#     for m in matches[:2]:
#         print("  ID:", m.get('id'), "|", m.get('text','')[:100])
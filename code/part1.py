"""PART 1: scikit-learn classifier vs foundation-model API.
Same task: binary ticket triage (ESCALATE vs MONITOR).
Measures accuracy, cost, latency.
"""
import pandas as pd, numpy as np, time, requests, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import os
API_KEY = os.environ.get('OPENROUTER_API_KEY', 'PASTE_YOUR_KEY_HERE')
MODEL = "openai/gpt-4o-mini"

df = pd.read_csv(r"C:\Users\Lenovo\Doubao\chats\2026-09-24\new-chat\A1\complaints.csv")
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============ (a) scikit-learn ============
t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
Xtr = vec.fit_transform(train_df["text"])
Xte = vec.transform(test_df["text"])
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, train_df["label"])
sklearn_train_time = time.time() - t0

t0 = time.time()
sklearn_pred = clf.predict(Xte)
sklearn_pred_time = time.time() - t0
sklearn_acc = accuracy_score(test_df["label"], sklearn_pred)
print(f"\n[sklearn] accuracy={sklearn_acc:.3f}  train_time={sklearn_train_time:.2f}s  predict_time={sklearn_pred_time:.4f}s")
print(classification_report(test_df["label"], sklearn_pred))

# ============ (b) Foundation model API (no training) ============
SYSTEM = (
    "You are a bank back-office triage assistant. Classify the customer ticket into exactly one label. "
    "ESCALATE = urgent: fraud, missing money, locked account, identity theft, frozen funds, legal threat. "
    "MONITOR = routine: card replacement, address update, fee explanation, statement request, password reset, branch hours. "
    "Reply with ONLY the label word, nothing else."
)

def api_predict(text):
    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":text}], "temperature":0},
        timeout=60,
    )
    dt = time.time() - t0
    d = r.json()
    label = d["choices"][0]["message"]["content"].strip()
    usage = d.get("usage", {})
    cost = usage.get("cost", 0)
    return label, dt, cost

api_preds, lats, costs = [], [], []
for i, row in test_df.iterrows():
    lab, lat, c = api_predict(row["text"])
    # normalise
    if "ESCALATE" in lab.upper():
        lab = "ESCALATE"
    elif "MONITOR" in lab.upper():
        lab = "MONITOR"
    else:
        lab = "UNKNOWN"
    api_preds.append(lab); lats.append(lat); costs.append(c)
    print(f"  [{len(api_preds)}/{len(test_df)}] pred={lab} gold={row['label']} lat={lat:.2f}s")

api_acc = accuracy_score(test_df["label"], api_preds)
api_total_cost = sum(costs)
api_avg_lat = np.mean(lats)
print(f"\n[API] accuracy={api_acc:.3f}  avg_latency={api_avg_lat:.2f}s  total_cost={api_total_cost:.6f}$")

# ============ Save comparison ============
results = {
    "sklearn": {"accuracy": round(sklearn_acc,3), "train_time_s": round(sklearn_train_time,2),
                "predict_per_sample_ms": round(sklearn_pred_time/len(test_df)*1000,3)},
    "api": {"accuracy": round(api_acc,3), "avg_latency_s": round(api_avg_lat,2),
            "total_cost_usd": round(api_total_cost,6), "cost_per_call": round(api_total_cost/len(test_df),8)},
    "n_test": len(test_df), "n_train": len(train_df),
}
with open(r"C:\Users\Lenovo\Doubao\chats\2026-09-24\new-chat\A1\part1_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", json.dumps(results, indent=2))



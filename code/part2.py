"""PART 2: Minimal RAG system over a small bank-policy document collection.
Build docs -> TF-IDF retrieve top-k chunks -> stuff into prompt -> LLM answers.
"""
import json, time, requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
API_KEY = os.environ.get('OPENROUTER_API_KEY', 'PASTE_YOUR_KEY_HERE')
MODEL = "openai/gpt-4o-mini"

# ---- 15 bank policy / FAQ documents (the "knowledge base") ----
DOCS = [
    ("Fee refund policy", "Customers may request a refund of account fees within 30 days of the charge. Refunds are discretionary and granted at the bank's sole judgement. Students on a basic account receive automatic fee waivers. Minimum balance fees are not refundable once the balance has remained below threshold for two consecutive billing cycles."),
    ("Unauthorized transaction claim", "Unauthorized transactions must be reported within 60 days of the statement date. Once reported, the customer is not liable for losses exceeding $100. The bank will complete its investigation within 10 business days and provisionally credit the account pending resolution. Reports made after 60 days may not be refunded."),
    ("Lost or stolen card", "Report a lost or stolen debit card immediately via the 24-hour hotline at 1800-123-4567. A replacement card arrives within 3 to 5 working days. Temporary lock is available in the mobile app. Transactions made before the report are the customer's responsibility; after the report the bank covers all losses."),
    ("ATM dispute", "If an ATM dispenses no cash but debits the account, customers should wait 24 hours for automatic reversal. If not reversed, file a dispute within 7 days. The investigation takes 5 to 10 business days. No evidence such as receipts is required for amounts under $200."),
    ("Account closure", "Personal accounts can be closed at any branch with photo ID. All scheduled payments must be cancelled first. If the account has a negative balance it must be settled before closure. Closure takes 3 working days. The bank may charge a $20 closure fee unless the account has been open for over 5 years."),
    ("Fixed deposit early withdrawal", "Breaking a fixed deposit before maturity incurs a penalty of 1 month's interest. The principal is never at risk. Early withdrawal is allowed once without penalty if the customer is facing financial hardship, subject to manager approval. Funds arrive in the current account within 2 working days."),
    ("Credit card chargeback", "Credit card chargebacks must be filed within 54 days of the transaction. The merchant is given 30 days to respond. Successful chargebacks refund the amount within 5 days. Disputes about quality of goods are not eligible for chargeback; customers must contact the merchant first."),
    ("Overseas card usage", "Debit cards work overseas in countries with Mastercard acceptance. A foreign transaction fee of 1.5% applies. Customers must notify the bank before travel to avoid the card being blocked for suspected fraud. Daily overseas withdrawal limit is $1,000."),
    ("Joint account rules", "Joint accounts operate on either-to-sign basis by default, meaning either holder can withdraw all funds. Both holders must be present to change the operating mode to all-to-sign. If one holder dies, the account freezes until probate is granted unless survivorship instructions are on file."),
    ("Loan repayment holiday", "Mortgage customers may apply for a repayment holiday of up to 6 months once every 3 years during the loan tenure. Interest still accrues during the holiday. Application must be made 30 days in advance. Approval is subject to current repayment history being clean for at least 12 months."),
    ("Account statement request", "Free e-statements are available in the mobile app for the past 7 years. Printed statements cost $5 per page and take 3 working days. Statements older than 7 years are archived and cost $20 and take 10 working days to retrieve."),
    ("Identity verification", "New customers must pass full identity verification using a government-issued photo ID and proof of address dated within 3 months. Existing customers changing address require only the new proof of address. Video verification is available for amounts under $50,000."),
    ("Data privacy complaint", "Customers may request their personal data held by the bank by submitting a Data Access Request form. The bank responds within 30 days. A fee of $20 applies. Customers may also request correction of inaccurate data free of charge. Complaints about data handling go to the PDPC."),
    ("Complaint escalation process", "Stage 1 complaints are handled by the service desk within 5 working days. If unresolved, customers may escalate to the customer relations manager within 30 days. Final unresolved matters may be referred to the Financial Industry Disputes Resolution Centre (FIDReC) free of charge."),
    ("Transaction limits", "Daily transfer limit via the app is set at $5,000 by default and can be raised to $20,000 in-app. Transfers above $20,000 require a branch visit. PayNow daily limit is $1,000 for new users and rises to $5,000 after 30 days. Bill payments are unlimited."),
]

# ---- Build index ----
names = [d[0] for d in DOCS]
texts = [d[1] for d in DOCS]
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(texts)

def retrieve(query, k=2):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [(names[i], texts[i], float(sims[i])) for i in idx]

def ask(query, k=2):
    hits = retrieve(query, k)
    context = "\n\n".join([f"[{n}] {t}" for n, t, s in hits])
    prompt = f"Answer the customer question using ONLY the bank policy context below. If the context does not contain the answer, say 'I cannot find this in our policy'. Be concise and cite the policy name.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type":"application/json"},
        json={"model": MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0}, timeout=60)
    return r.json()["choices"][0]["message"]["content"], hits

# ---- 5 questions that NEED the docs ----
questions = [
    "How long do I have to report an unauthorized transaction?",
    "What is the fee for requesting a printed statement older than 7 years?",
    "My mortgage repayment holiday - how many times can I take one and how long can it last?",
    "I need to dispute an ATM that didn't give me cash. How long should I wait before filing?",
    "What happens if I break my fixed deposit early - is my principal safe?",
]

results = []
print("=== 5 WORKING ANSWERS ===")
for q in questions:
    ans, hits = ask(q)
    print(f"\nQ: {q}")
    print(f"  Retrieved: {[h[0] for h in hits]}")
    print(f"  A: {ans}")
    results.append({"q": q, "retrieved":[h[0] for h in hits], "answer": ans})

# ---- 1 BREAK IT: question whose correct doc is semantically tricky ----
# "I lost my card while travelling overseas and someone charged my account. Am I liable?"
# This needs BOTH the lost-card doc AND the overseas doc. Try k=1 so it only grabs one.
print("\n\n=== BREAKING THE SYSTEM ===")
trick_q = "I lost my debit card on holiday and someone spent $800 on it at a shop abroad. Am I liable, and was I supposed to tell the bank before travelling?"
ans1, hits1 = ask(trick_q, k=1)
print(f"Q: {trick_q}")
print(f"  Retrieved (k=1): {[h[0] for h in hits1]}")
print(f"  A: {ans1}")

# Save
out = {"working_qa": results,
       "failure": {"q": trick_q, "retrieved_k1":[h[0] for h in hits1], "answer": ans1}}
with open(r"C:\Users\Lenovo\Doubao\chats\2026-09-24\new-chat\A1\part2_results.json","w") as f:
    json.dump(out, f, indent=2)
print("\nSaved part2 results")



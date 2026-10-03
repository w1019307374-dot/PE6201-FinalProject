"""PART 3 v2: harder eval set + deliberately weak v1 prompt.
v1 = vague, no format, no definitions. v2 = structured prompt with definitions.
"""
import requests, json

import os
API_KEY = os.environ.get('OPENROUTER_API_KEY', 'PASTE_YOUR_KEY_HERE')
MODEL = "openai/gpt-4o-mini"

# Harder, ambiguous cases
EVAL = [
    {"input":"Someone used my card without permission and spent $3000", "expected":"ESCALATE"},
    {"input":"I would like a replacement debit card", "expected":"MONITOR"},
    {"input":"$800 disappeared from my account overnight", "expected":"ESCALATE"},
    {"input":"Can you explain the $2 monthly fee on my statement?", "expected":"MONITOR"},
    {"input":"My account is locked and I cannot access my salary", "expected":"ESCALATE"},
    {"input":"I got a phone call from someone saying they are from your fraud department asking for my OTP. Is that normal?", "expected":"MONITOR"},
    {"input":"There is a $12 charge on my card I do not recognise, probably a free trial that converted", "expected":"MONITOR"},
    {"input":"I want to close my account and transfer my balance to another bank", "expected":"MONITOR"},
    {"input":"A cheque I deposited bounced and the payee is threatening to take me to court", "expected":"ESCALATE"},
    {"input":"Help me update my mailing address on file", "expected":"MONITOR"},
]

# v1: deliberately weak - no format, no definitions, no examples
PROMPT_V1 = "Triage this customer ticket."

# v2: structured, with definitions and JSON schema
PROMPT_V2 = (
    'You are a bank triage assistant. Classify the ticket and reply ONLY as JSON: '
    '{"label": "ESCALATE" or "MONITOR", "reason": "one short sentence"}. '
    "ESCALATE = active fraud, money missing without explanation, locked/frozen funds, or legal/collection threat. "
    "MONITOR = routine requests, inquiries, or small disputed charges that can go through normal process. "
    "If the customer is merely ASKING whether something is normal, that is MONITOR even if they mention fraud."
)

def run_prompt(prompt, ticket):
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type":"application/json"},
        json={"model":MODEL, "messages":[
            {"role":"system","content":prompt},
            {"role":"user","content":ticket}], "temperature":0}, timeout=60)
    return r.json()["choices"][0]["message"]["content"]

def l1_assertions(raw, expected):
    checks = {}
    try:
        obj = json.loads(raw)
        checks["valid_json"] = True
    except Exception:
        checks["valid_json"] = False
        return False, checks
    checks["has_label_field"] = "label" in obj
    checks["label_in_vocab"] = obj.get("label") in ("ESCALATE","MONITOR")
    checks["label_correct"] = obj.get("label") == expected
    return all(checks.values()), checks

def l2_judge(ticket, raw, expected):
    judge = (f"Grade this triage. Ticket: '{ticket}'. Output: {raw}. Expected: {expected}. "
             'Reply JSON only: {"correct": true/false}. correct=true means the label is right AND a supervisor could use it.')
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type":"application/json"},
        json={"model":MODEL, "messages":[{"role":"user","content":judge}], "temperature":0}, timeout=60)
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"]).get("correct", False)
    except:
        return False

def evaluate(prompt, name):
    rows=[]; l1p=0; l2p=0
    for c in EVAL:
        raw = run_prompt(prompt, c["input"])
        a, chk = l1_assertions(raw, c["expected"])
        b = l2_judge(c["input"], raw, c["expected"])
        l1p += a; l2p += b
        rows.append({"input":c["input"],"expected":c["expected"],"raw":raw,"l1":chk,"l1_pass":a,"l2_pass":b})
        print(f"  [{name}] {c['input'][:45]:45s} L1={'PASS' if a else 'FAIL'} L2={'PASS' if b else 'FAIL'} | {raw[:60]}")
    n=len(EVAL)
    return {"n":n,"l1_pass_rate":l1p/n,"l2_pass_rate":l2p/n,"rows":rows}

print("=== V1 (weak prompt) ===")
v1 = evaluate(PROMPT_V1, "v1")
print(f"V1: L1={v1['l1_pass_rate']*100:.0f}%  L2={v1['l2_pass_rate']*100:.0f}%")

print("\n=== V2 (structured prompt) ===")
v2 = evaluate(PROMPT_V2, "v2")
print(f"V2: L1={v2['l1_pass_rate']*100:.0f}%  L2={v2['l2_pass_rate']*100:.0f}%")

out={"v1":v1,"v2":v2,"l1_delta":round(v2['l1_pass_rate']-v1['l1_pass_rate'],2),
     "l2_delta":round(v2['l2_pass_rate']-v1['l2_pass_rate'],2)}
with open(r"C:\Users\Lenovo\Doubao\chats\2026-09-24\new-chat\A1\part3_results.json","w") as f:
    json.dump(out,f,indent=2)
print(f"\nL1 delta: {out['l1_delta']*100:+.0f}pp  L2 delta: {out['l2_delta']*100:+.0f}pp")



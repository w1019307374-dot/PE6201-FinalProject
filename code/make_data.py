"""Generate a realistic labelled bank-complaint dataset for A1.
Labels: ESCALATE (urgent, duty supervisor must review) vs MONITOR (routine).
We build templates with variation so the scikit-learn model has something to learn.
"""
import random, json, csv

random.seed(42)

# ---- ESCALATE templates: fraud, missing money, locked account, legal ----
escalate_templates = [
    "I just checked my account and there is an unauthorized transaction of {amt} that I did not make. Someone must have used my card. This is fraud.",
    "Money has gone missing from my account. I had {amt} in my balance this morning and now it is gone. I need this investigated immediately.",
    "My account has been locked and I cannot access my salary. I need urgent help to release my funds today.",
    "I was a victim of identity theft. Someone opened an account in my name and took out a loan. This needs to be escalated to your fraud team.",
    "Your bank wrongly closed my business account overnight and froze all my operating funds. I cannot pay my staff. This is urgent.",
    "There is a suspicious standing instruction I never set up that is transferring {amt} every week to an unknown account. Please stop it now.",
    "I think my online banking was hacked. There are login attempts from a foreign country and I received an OTP I did not request.",
    "A cheque I deposited was bounced without explanation and now the payee is threatening legal action against me. I need a manager to call me today.",
    "My card was cloned and used at {loc} for {amt}. I have already reported it but no one is refunding my money.",
    "I received a call from someone claiming to be from your fraud department asking for my PIN. I think I was scammed and now {amt} is missing.",
    "Your ATM dispensed no cash but deducted {amt} from my account. I have been waiting three days and nobody is helping.",
    "I am being overcharged by a subscription I never authorised and the amount is {amt}. This needs a refund urgently.",
]

monitor_templates = [
    "I would like to request a replacement debit card as my current one is scratched and the chip does not read.",
    "Could you please help me update my mailing address on file? I moved house last month.",
    "I noticed a {fee} administrative fee on my statement. Could you explain what this charge is for?",
    "I need a copy of my account statement for the past six months for my landlord.",
    "I forgot my online banking password and would like guidance on how to reset it myself.",
    "Can you tell me the current interest rate on my savings account?",
    "I would like to order a new chequebook. How long does it take to arrive?",
    "My credit card statement is not showing in the app. Could you refresh it on your end?",
    "I want to set up a recurring transfer to my savings account on payday. Is this something I can do in the app?",
    "Could you please waive the monthly account fee for this month as I am a student?",
    "I received a marketing email about a credit card offer. How do I opt out of these emails?",
    "I would like to know your branch opening hours on Saturday.",
]

amts = ["$480", "$1,200", "$3,500", "$850", "$12,400", "$2,100", "$6,750", "$320", "$15,000"]
locs = ["a petrol station", "a shopping mall", "another city", "an online store", "a restaurant"]
fees = ["$2", "$5", "$1.50", "$3", "$10"]

def make_one(tpl, label):
    t = random.choice(tpl) if isinstance(tpl, list) else tpl
    return t.format(
        amt=random.choice(amts),
        loc=random.choice(locs),
        fee=random.choice(fees),
    )

rows = []
for i in range(200):
    rows.append((make_one(escalate_templates, "ESCALATE"), "ESCALATE"))
for i in range(200):
    rows.append((make_one(monitor_templates, "MONITOR"), "MONITOR"))

random.shuffle(rows)

with open(r"C:\Users\Lenovo\Doubao\chats\2026-09-24\new-chat\A1\complaints.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    for text, label in rows:
        w.writerow([text, label])

print(f"Wrote {len(rows)} rows")
print("Sample ESCALATE:", rows[0][0][:100])



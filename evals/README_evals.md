# Eval explainer

## What this is
`eval_cases.json` — 10 held-out tickets used by `code/part3.py`. Deliberately mixes obvious
cases with **ambiguous** ones, because obvious cases pass every prompt and teach you nothing.

## Two ambiguous cases (on purpose)
- Case 6: customer merely **asks** if a fraud-department call is normal → expected MONITOR.
  The word "fraud" appears but this is not an active incident.
- Case 7: a small $12 unrecognised charge → expected MONITOR, not ESCALATE.

## Two-level scoring
- **L1 (automatic assertions, machine-checkable):** valid JSON, `label` field present,
  label in {ESCALATE, MONITOR}, label matches expected.
- **L2 (judgement, LLM-as-judge):** a second gpt-4o-mini call scores whether the triage is
  actually right and usable by a supervisor.

## Result (measured, not estimated)
| prompt | L1 pass | L2 pass |
|---|---|---|
| v1 = "Triage this customer ticket." (deliberately weak) | 0% | 100% |
| v2 = JSON schema + category definitions | 100% | 100% |

v1 failed L1 on every case because the model returned markdown prose, not JSON — even though
its triage decisions were all correct (L2=100%). This is the key finding: a weak prompt can be
*semantically right but integration-broken*.

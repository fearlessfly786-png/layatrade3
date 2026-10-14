"""Run the real Laya decision model (downloads checkpoint from HuggingFace on first use)."""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")

import time
import laya

print("Loading checkpoint (downloads ~1.7GB on first run)...", flush=True)
t0 = time.time()
agent = laya.load("convaiinnovations/laya")
print(f"Loaded in {time.time() - t0:.1f}s\n")

# Test 1: the official hello-world decision
result = agent.predict(
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
    {"refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    }},
)
answer = result["answers"]["refund_requested"]
print("TEST 1 (refund detection)")
print("  P(refund requested) = %.4f  (confidence %.4f)" % (answer["noul"], answer["confidence"]))
print("  FULL:", answer, "\n")

# Test 2: choice question — which department should handle this?
result2 = agent.predict(
    {"body": "The app crashes every time I open settings since yesterday's update. Nothing loads."},
    {"department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    }},
)
print("TEST 2 (department routing)")
print("  Full:", result2["answers"]["department"], "\n")

# Test 3: score question — urgency
result3 = agent.predict(
    {"body": "Our whole team cannot log in and we are losing customers right now."},
    {"urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "blocking"],
    }},
)
print("TEST 3 (urgency scoring)")
print("  Full:", result3["answers"]["urgency"])

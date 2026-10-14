"""Held-out evaluation of the fine-tuned trading checkpoint.

Evaluates laya-trading-final on:
  1. trade_eval.jsonl   (994 rows, REAL market distribution — the honest headline number)
  2. eval_balanced.jsonl (150 rows, 50/class — per-class fair view)

Metrics:
  - action (choice): accuracy, per-class precision/recall/F1, confusion matrix
  - conviction (score): exact match, MAE, within-1
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import laya

CKPT = "./laya-trading-final"
QUESTIONS = json.load(open(os.path.join(CKPT, "questions.json")))


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def predict_all(agent, rows, label):
    """Predict in chunks; returns (actions, convictions) lists aligned with rows."""
    actions, convictions = [], []
    t0 = time.time()
    CH = 100
    for i in range(0, len(rows), CH):
        chunk = rows[i:i + CH]
        states = [r["state"] for r in chunk]
        results = agent.predict_batch(states, QUESTIONS, batch_size=8)
        for res in results:
            ans = res["answers"]
            a = ans["action"]
            c = ans["conviction"]
            actions.append({
                "choice": a.get("choice"),
                "probs": a.get("probabilities", {}),
            })
            convictions.append({
                "score": c.get("score"),
                "probs": c.get("probabilities", {}),
            })
        done = min(i + CH, len(rows))
        el = time.time() - t0
        print(f"  [{label}] {done}/{len(rows)} states ({el:.0f}s)", flush=True)
    return actions, convictions


def eval_action(actions, rows):
    labels = ["buy", "hold", "sell"]
    y_true = [r["expected"]["action"] for r in rows]
    y_pred = [a["choice"] for a in actions]
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n

    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[o][c] for o in labels if o != c)
        fn = sum(cm[c][o] for o in labels if o != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for t in y_true if t == c)}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)
    return acc, per_class, cm, macro_f1


def eval_conviction(convictions, rows):
    y_true = [float(r["expected"]["conviction"]) for r in rows]
    n = len(y_true)
    exact = within1 = 0
    abs_err_sum = 0.0
    n_valid = 0
    for c, t in zip(convictions, y_true):
        s = c["score"]
        if s is None:
            continue
        s = float(s)
        # score may be continuous (expected value) — round for match metrics
        sr = round(s)
        exact += (sr == t)
        within1 += (abs(sr - t) <= 1)
        abs_err_sum += abs(s - t)
        n_valid += 1
    return {
        "n": n_valid,
        "exact_match": exact / n_valid if n_valid else None,
        "within_1": within1 / n_valid if n_valid else None,
        "mae": abs_err_sum / n_valid if n_valid else None,
    }


def show_sample(actions, convictions):
    print("  sample action answer:", json.dumps(actions[0])[:220])
    print("  sample conviction answer:", json.dumps(convictions[0])[:220])


def run(agent, path, label, sample=False):
    rows = load_rows(path)
    print(f"\n=== {label}: {path} ({len(rows)} rows) ===", flush=True)
    actions, convictions = predict_all(agent, rows, label)
    if sample:
        show_sample(actions, convictions)

    acc, per_class, cm, macro_f1 = eval_action(actions, rows)
    print(f"\n  ACTION accuracy: {acc:.4f}   macro-F1: {macro_f1:.4f}")
    print(f"  {'class':<6} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>8}")
    for c in ("buy", "hold", "sell"):
        v = per_class[c]
        print(f"  {c:<6} {v['precision']:>7.4f} {v['recall']:>7.4f} {v['f1']:>7.4f} {v['support']:>8}")
    print("  confusion matrix (rows=true, cols=pred):")
    print(f"  {'':<8} {'buy':>7} {'hold':>7} {'sell':>7}")
    for t in ("buy", "hold", "sell"):
        print(f"  {t:<8} {cm[t]['buy']:>7} {cm[t]['hold']:>7} {cm[t]['sell']:>7}")

    conv = eval_conviction(convictions, rows)
    print(f"  CONVICTION: exact={conv['exact_match']:.4f} within1={conv['within_1']:.4f} mae={conv['mae']:.4f}")

    # mean probability mass the model puts on the true action class
    mass = []
    for r, a in zip(rows, actions):
        p = a["probs"].get(r["expected"]["action"])
        if p is not None:
            mass.append(float(p))
    if mass:
        print(f"  mean P(true action) = {sum(mass)/len(mass):.4f}")

    return {
        "dataset": label, "n": len(rows), "action_accuracy": acc,
        "macro_f1": macro_f1, "per_class": per_class, "confusion": cm,
        "conviction": conv,
    }


def main():
    print("Loading checkpoint...", flush=True)
    t0 = time.time()
    agent = laya.load(CKPT)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    results = []
    # smoke + full run on the real-distribution eval set first
    results.append(run(agent, "trade_eval.jsonl", "REAL-DISTRIBUTION HELD-OUT", sample=True))
    results.append(run(agent, "eval_balanced.jsonl", "BALANCED HELD-OUT (50/class)"))

    with open("final_eval_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved -> final_eval_results.json", flush=True)


if __name__ == "__main__":
    main()

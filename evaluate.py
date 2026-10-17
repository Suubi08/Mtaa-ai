"""Measure the prototype on the labelled sample reports.

Run:  python evaluate.py
Outputs: category accuracy, valid structured output rate, duplicate precision/recall,
and average processing time per report. These are prototype test results on a
tiny hand-labelled set, not proof of real-world impact.
"""
import itertools
import json
import os
import time

os.environ["MTAAWATCH_DB"] = "eval_tmp.db"
import core  # noqa: E402

with open("sample_reports.json") as f:
    samples = json.load(f)

times, results = [], []
for s in samples:
    t0 = time.time()
    a = core.analyze_report(s["description"], s["location"])
    times.append(time.time() - t0)
    results.append({**s, **{"pred": a["category"], "source": a["source"], "summary_p": a["summary"]}})

acc = sum(r["pred"] == r["category"] for r in results) / len(results)
valid = sum(r["source"] == "gemma" for r in results) / len(results)

# Duplicate evaluation over all pairs; true duplicate = same group label.
tp = fp = fn = 0
for x, y in itertools.combinations(results, 2):
    a = {"description": x["description"], "location": x["location"], "category": x["pred"], "summary": x["summary_p"]}
    b = {"description": y["description"], "location": y["location"], "category": y["pred"], "summary": y["summary_p"]}
    flagged = core.duplicate_score(a, b) >= 0.4
    truth = x["group"] == y["group"]
    tp += flagged and truth
    fp += flagged and not truth
    fn += (not flagged) and truth

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0

print(f"Reports tested:              {len(results)}")
print(f"Category accuracy:           {acc:.0%}")
print(f"Valid Gemma outputs:         {valid:.0%} (rest used keyword fallback)")
print(f"Duplicate precision:         {precision:.0%}  (tp={tp}, fp={fp})")
print(f"Duplicate recall:            {recall:.0%}  (fn={fn})")
print(f"Avg time per analysis:       {sum(times) / len(times):.2f}s")
for r in results:
    mark = "ok " if r["pred"] == r["category"] else "BAD"
    print(f"  [{mark}] #{r['id']} expected={r['category']} got={r['pred']} via {r['source']}")

if os.path.exists("eval_tmp.db"):
    os.remove("eval_tmp.db")

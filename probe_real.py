"""Run the app's real analysis on sample reports with a long timeout and show timings.
Run: python probe_real.py            (uses your configured model)
     python probe_real.py gemma-4-31b-it   (try another model)"""
import sys
import time

import core

core.TIMEOUT = 120  # generous, so we can see how long the model really needs
if len(sys.argv) > 1:
    core.MODEL = sys.argv[1]
print(f"Model: {core.MODEL}\n")

tests = [
    ("The drain near the market has been blocked for days and water is collecting on the road.", "Kasubi Market"),
    ("Omuti gwa masanyalaze gwa gude mukubo", "Mukono"),
]
for desc, loc in tests:
    t = time.time()
    r = core.analyze_report(desc, loc)
    print(f"REPORT: {desc}")
    print(f"  source={r['source']}  total={time.time() - t:.1f}s  api={core.LAST_CALL}")
    print(f"  category={r['category']}  urgency={r['urgency']}  language={r['language']}")
    print(f"  summary={r['summary']}")
    if r.get("error"):
        print(f"  ERROR: {r['error']}")
    print()

"""Quick check that the AI connection works. Run: python check_ai.py"""
import core

print(f"Provider: {core.PROVIDER}   Model: {core.MODEL}")
print(f"API key found: {'yes' if core.API_KEY else 'no'}")

tests = [
    ("The drain near the market has been blocked for days and water is collecting on the road.", "Kasubi Market"),
    ("Rubbish has piled up at the trading centre.", "Nakawa"),
]
for desc, loc in tests:
    r = core.analyze_report(desc, loc)
    print("\nREPORT:", desc)
    print("  source:", r["source"], "| category:", r["category"], "| urgency:", r["urgency"], "| language:", r["language"])
    print("  summary:", r["summary"])
    if r.get("error"):
        print("  ERROR:", r["error"])

print("\nIf source says 'gemma', the API call works. If 'fallback', read the ERROR line above.")

"""Diagnose the Gemma API: lists available Gemma models and times a tiny request.
Run: python probe_api.py   (and paste the output if something fails)"""
import time

import core
from google import genai
from google.genai import types

print(f"Model: {core.MODEL}   API key found: {'yes' if core.API_KEY else 'no'}")
client = genai.Client(api_key=core.API_KEY, http_options=types.HttpOptions(timeout=30000))

print("\nGemma models available to your key:")
try:
    for m in client.models.list():
        if "gemma" in m.name.lower():
            print("  ", m.name)
except Exception as e:
    print("   could not list models:", str(e)[:200])

for label, extra in [("default", {}),
                     ("thinking off", {"thinking_config": types.ThinkingConfig(thinking_budget=0)})]:
    t = time.time()
    try:
        r = client.models.generate_content(
            model=core.MODEL,
            contents='Reply with only this JSON: {"ok": true}',
            config=types.GenerateContentConfig(max_output_tokens=512, **extra),
        )
        u = r.usage_metadata
        print(f"\n[{label}] {time.time() - t:.1f}s  thinking tokens={getattr(u, 'thoughts_token_count', None)}"
              f"  answer tokens={getattr(u, 'candidates_token_count', None)}")
        print("   text:", repr((r.text or "")[:200]))
    except Exception as e:
        print(f"\n[{label}] FAILED after {time.time() - t:.1f}s: {str(e)[:250]}")

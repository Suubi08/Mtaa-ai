# MtaaWatch AI 🇺🇬

Community reporting and civic issue prioritization for Uganda. Residents describe a problem in plain language (English, Luganda or Swahili), optionally with a photo. Gemma turns it into a structured report, duplicates are flagged, and a human reviewer merges them into organized cases.

**Design rule:** Gemma does language understanding only. Validation, duplicate scoring, department mapping and storage are deterministic code. Location comes from the resident, never from the model. Nothing is sent to any authority, and no merge happens without a human clicking "Same incident".

## Run it (Gemma through the Gemini API)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set it up:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste your key into .env
python check_ai.py              # should print: source: gemma
streamlit run app.py
```

The default model is `gemma-4-26b-a4b-it`. Change it with `MTAAWATCH_MODEL` if Google AI Studio lists a different Gemma name.
To run a local model instead, `pip install ollama` and set `MTAAWATCH_PROVIDER=ollama`.
If the API is unreachable, the app falls back to a keyword classifier and labels those reports `fallback`.

## Demo flow

1. Sidebar → **Load 10 sample reports**.
2. **Review queue** → confirm or reject suspected duplicates.
3. **Dashboard** → raw reports vs. organized cases, by category, location and status.

## Measure it

```bash
python evaluate.py
```

Reports category accuracy, valid structured-output rate, duplicate precision/recall and average time per report on `sample_reports.json`. These are prototype results on a tiny hand-labelled set, not evidence of community-wide impact.

## Privacy and limits

- With the API, the report text and typed location are sent to Google. Photos stay on the local machine and are never sent to the model.
- Photos are re-encoded on upload (metadata removed) and stored in `photos/`. The SQLite file is local and the app has no accounts, so do not expose it publicly.
- Keep your API key in `.env` only. Never paste it into code or commit it.
- Urgency is a preliminary flag, and Luganda understanding should be checked with native speakers.

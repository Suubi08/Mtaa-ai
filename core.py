"""MtaaWatch AI core logic: storage, Gemma analysis, duplicate detection.

Design rule: Gemma only handles language understanding. Validation,
similarity scoring and storage are deterministic code, and a human always
makes the final merge decision.
"""
import json
import os
import re
import sqlite3
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # optional: load GEMINI_API_KEY etc. from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
           or os.getenv("API_KEY"))
# Provider: "gemini" = Gemma hosted by Google (API call), "ollama" = local model.
PROVIDER = os.getenv("MTAAWATCH_PROVIDER") or ("gemini" if API_KEY else "ollama")
MODEL = os.getenv("MTAAWATCH_MODEL") or (
    "gemma-4-26b-a4b-it" if PROVIDER == "gemini" else "gemma4:e2b")
TIMEOUT = float(os.getenv("MTAAWATCH_TIMEOUT", "30"))
MAX_TOKENS = int(os.getenv("MTAAWATCH_MAX_TOKENS", "2048"))
FAST_INSTRUCTION = ("Answer immediately with only the requested JSON object. "
                    "Do not reason step by step and do not add explanations.")
LAST_CALL = {}  # timing of the most recent API call, used by probe_real.py
DB = os.getenv("MTAAWATCH_DB", "mtaawatch.db")
PHOTO_DIR = os.getenv("MTAAWATCH_PHOTOS", "photos")
MAX_PHOTO_BYTES = 8 * 1024 * 1024


def _extract_json(text):
    """Parse a JSON object from model text, tolerating code fences or extra prose."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in model output")
        return json.loads(text[start:end + 1])


def _gemini_call(client, types, prompt, thinking_off):
    cfg = {"temperature": 0, "max_output_tokens": MAX_TOKENS,
           "system_instruction": FAST_INSTRUCTION}
    if thinking_off:
        # Our task is a short classification: skip slow "thinking" where supported.
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return client.models.generate_content(
        model=MODEL, contents=prompt, config=types.GenerateContentConfig(**cfg))


def _generate(prompt):
    """Send the prompt to the configured provider and return the raw text."""
    if PROVIDER == "gemini":
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai
        from google.genai import types
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=int(TIMEOUT * 1000)),
        )
        # gemma-4-26b-a4b-it rejects "thinking budget" (HTTP 400), so it is off by default.
        thinking_off = os.getenv("MTAAWATCH_DISABLE_THINKING") == "1"
        last_exc = None
        t0 = time.time()
        for _ in range(3):
            try:
                resp = _gemini_call(client, types, prompt, thinking_off)
                break
            except Exception as e:
                last_exc = e
                msg = str(e).upper()
                if thinking_off and ("THINKING" in msg or "INVALID_ARGUMENT" in msg):
                    thinking_off = False  # this model rejects the setting; retry without it
                    continue
                if any(k in msg for k in ("503", "504", "429", "DEADLINE", "UNAVAILABLE", "OVERLOADED")):
                    time.sleep(1.5)  # temporary server trouble; try again
                    continue
                raise
        else:
            raise last_exc
        try:
            LAST_CALL.update(seconds=round(time.time() - t0, 1),
                             thinking_tokens=getattr(resp.usage_metadata, "thoughts_token_count", None))
        except Exception:
            pass
        text = resp.text or ""
        if not text.strip():
            try:
                reason = resp.candidates[0].finish_reason
            except Exception:
                reason = "unknown"
            raise RuntimeError(f"empty model response (finish_reason={reason})")
        return text
    import ollama
    client = ollama.Client(timeout=TIMEOUT)
    resp = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 300},
        keep_alive=os.getenv("MTAAWATCH_KEEP_ALIVE", "2m"),
    )
    return resp["message"]["content"]

CATEGORIES = [
    "blocked_drainage",
    "uncollected_waste",
    "damaged_road",
    "water_point",
    "power_hazard",
    "other",
]
URGENCY = ["low", "medium", "high", "unknown"]

# Deterministic mapping: the model never decides who is responsible.
DEPARTMENTS = {
    "blocked_drainage": "Works / Drainage (Division or Town Council)",
    "uncollected_waste": "Public Health / Solid Waste Management",
    "damaged_road": "Roads and Works (Division or Town Council)",
    "water_point": "Water and Sanitation (water office)",
    "power_hazard": "Electricity distributor and local authority (urgent safety issue; confirm the right contact)",
    "other": "Local administration (needs triage)",
}

# Used only if Gemma is unreachable or returns unusable output.
KEYWORDS = {
    # Luganda terms are a small starter set: have a native speaker verify/extend them.
    "blocked_drainage": ["drain", "drainage", "channel", "flood", "water is collecting", "sewer", "culvert",
                         "omwala", "emyala"],
    "uncollected_waste": ["rubbish", "garbage", "waste", "trash", "dump", "litter", "skip",
                          "kasasiro", "ebisasiro"],
    "damaged_road": ["pothole", "potholes", "bridge", "tarmac", "murram", "damaged road", "road is damaged",
                     "ekinnya", "ebinnya"],
    "water_point": ["tap", "borehole", "water point", "standpipe", "pump", "well"],
    "power_hazard": ["electric", "power line", "live wire", "transformer", "masanyalaze",
                     "amasannyalaze", "amasanyalaze"],
}
HIGH_WORDS = ["children", "school", "hospital", "disease", "cholera", "accident", "injur",
              "flood", "danger", "overflow", "collapsed", "sewage",
              "electric", "live wire", "masanyalaze", "amasannyalaze", "amasanyalaze"]
MEDIUM_WORDS = ["days", "weeks", "smell", "mosquito", "blocked", "pile"]


def connect():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                location TEXT NOT NULL,
                summary TEXT NOT NULL,
                urgency TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                department TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'gemma',
                status TEXT NOT NULL DEFAULT 'Needs review',
                duplicate_of INTEGER,
                dup_score REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                language TEXT NOT NULL DEFAULT 'unknown',
                photo_path TEXT,
                ai_error TEXT
            )
        """)
        # Migrate databases created before the language column existed.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(incidents)").fetchall()]
        if "language" not in cols:
            conn.execute("ALTER TABLE incidents ADD COLUMN language TEXT NOT NULL DEFAULT 'unknown'")
        if "photo_path" not in cols:
            conn.execute("ALTER TABLE incidents ADD COLUMN photo_path TEXT")
        if "ai_error" not in cols:
            conn.execute("ALTER TABLE incidents ADD COLUMN ai_error TEXT")


def save_photo(data):
    """Validate and store an uploaded photo. Returns a file path or None.

    The image is decoded and re-encoded as a resized JPEG under a random
    name, so the original filename and hidden metadata (e.g. GPS in EXIF)
    are not kept, and non-image files are rejected.
    """
    if not data or len(data) > MAX_PHOTO_BYTES:
        return None
    try:
        import io
        import uuid
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((1280, 1280))
        os.makedirs(PHOTO_DIR, exist_ok=True)
        path = os.path.join(PHOTO_DIR, f"{uuid.uuid4().hex}.jpg")
        img.save(path, "JPEG", quality=85)
        return path
    except Exception:
        return None


# ---------------------------------------------------------------- analysis
def _fallback_analyze(description, location):
    text = description.lower()
    scores = {c: sum(k in text for k in kws) for c, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    category = best if scores[best] > 0 else "other"
    if any(w in text for w in HIGH_WORDS):
        urgency = "high"
    elif any(w in text for w in MEDIUM_WORDS):
        urgency = "medium"
    else:
        # Unrecognised text must not look "low risk": we simply don't know.
        urgency = "unknown" if category == "other" else "low"
    return {
        "category": category,
        "location": (location or "").strip(),
        "summary": description.strip()[:140],
        "urgency": urgency,
        "reason": "Keyword fallback used (model unavailable or output invalid).",
        "source": "fallback",
        "language": "unknown",
    }


def validate_analysis(raw, description, location):
    """Return a clean dict or raise ValueError. Never trust model output."""
    if not isinstance(raw, dict):
        raise ValueError("not an object")
    category = str(raw.get("category", "")).strip().lower()
    if category not in CATEGORIES:
        raise ValueError(f"bad category: {category}")
    urgency = str(raw.get("urgency", "unknown")).strip().lower()
    if urgency not in URGENCY:
        urgency = "unknown"
    summary = str(raw.get("summary", "")).strip()
    if not summary:
        raise ValueError("empty summary")
    # Location comes from the person, not the model, so it cannot be invented.
    return {
        "category": category,
        "location": (location or "").strip(),
        "summary": summary[:300],
        "urgency": urgency,
        "reason": str(raw.get("reason", "")).strip()[:300],
        "source": "gemma",
        "language": (str(raw.get("language", "")).strip()[:30] or "unknown"),
    }


def build_prompt(description, location=""):
    return f"""Classify this community report from Uganda. It may be in English, Luganda, Swahili or a mix.
Reply with ONLY a JSON object with the keys: category, urgency, language, summary, reason.
- category: one of {CATEGORIES}
- urgency: one of {URGENCY}. Use "high" only for clear immediate danger (health risk, flooding, injury, a fallen electric pole or live wires). It is a preliminary flag, not a verified assessment.
- language: main language of the report, e.g. "English", "Luganda", "Swahili"
- summary: one short sentence IN ENGLISH using only facts in the report. If a word is unclear, leave it out instead of guessing.
- reason: one short sentence in English explaining the urgency.
Never invent facts or claim any authority was notified.

Report: {description}
Reported location: {location or "Not provided"}"""


def analyze_report(description, location=""):
    """Ask Gemma for a structured report; fall back safely on any failure."""
    prompt = build_prompt(description, location)
    try:
        text = _generate(prompt)
        try:
            raw = _extract_json(text)
        except ValueError as e:
            # Keep a short sample of what the model said, to make problems diagnosable.
            raise ValueError(f"{e}; model said: {text[:250]!r}")
        return validate_analysis(raw, description, location)
    except Exception as e:
        result = _fallback_analyze(description, location)
        result["error"] = f"{type(e).__name__}: {e}"[:300]
        return result


# -------------------------------------------------------------- duplicates
def _norm(text):
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _sim(a, b, analyzer="word"):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 1) if analyzer == "word" else (2, 4))
        m = vec.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


def duplicate_score(new, old):
    """Compare description and location separately, then combine.

    Different categories are never flagged. Missing location relies on
    description alone but with a lower ceiling so uncertain cases stay
    low-confidence.
    """
    if new["category"] != old["category"]:
        return 0.0
    desc = max(_sim(new["summary"], old["summary"]),
               _sim(new["description"], old["description"]))
    if _norm(new["location"]) and _norm(old["location"]):
        loc = _sim(new["location"], old["location"], analyzer="char_wb")
        return round(0.5 * desc + 0.5 * loc, 3)
    return round(0.6 * desc, 3)


def find_duplicates(new, threshold=0.4, limit=3):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE duplicate_of IS NULL AND status != 'Rejected'"
        ).fetchall()
    scored = [(duplicate_score(new, dict(r)), dict(r)) for r in rows]
    scored = [s for s in scored if s[0] >= threshold]
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:limit]


# ----------------------------------------------------------------- storage
def save_incident(description, analysis):
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO incidents
               (description, category, location, summary, urgency, reason,
                department, source, language, photo_path, ai_error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (description, analysis["category"], analysis["location"],
             analysis["summary"], analysis["urgency"], analysis["reason"],
             DEPARTMENTS[analysis["category"]], analysis["source"],
             analysis.get("language", "unknown"), analysis.get("photo_path"),
             analysis.get("error")),
        )
        return cur.lastrowid


def set_duplicate(incident_id, original_id, score):
    """Record a suspected duplicate. A human must confirm it."""
    with connect() as conn:
        conn.execute(
            "UPDATE incidents SET duplicate_of=?, dup_score=?, status='Possible duplicate' WHERE id=?",
            (original_id, score, incident_id),
        )


def confirm_duplicate(incident_id):
    with connect() as conn:
        conn.execute("UPDATE incidents SET status='Merged' WHERE id=?", (incident_id,))


def reject_duplicate(incident_id):
    with connect() as conn:
        conn.execute(
            "UPDATE incidents SET duplicate_of=NULL, dup_score=NULL, status='Needs review' WHERE id=?",
            (incident_id,),
        )


def set_status(incident_id, status):
    with connect() as conn:
        conn.execute("UPDATE incidents SET status=? WHERE id=?", (status, incident_id))


def submit_report(description, location="", photo_bytes=None):
    """Full pipeline. Returns (incident_id, analysis, matches, seconds).

    analysis["photo_path"] is set only if a valid photo was stored.
    """
    t0 = time.time()
    analysis = analyze_report(description, location)
    analysis["photo_path"] = save_photo(photo_bytes) if photo_bytes else None
    new = {"description": description, **analysis}
    matches = find_duplicates(new)
    incident_id = save_incident(description, analysis)
    if matches:
        set_duplicate(incident_id, matches[0][1]["id"], matches[0][0])
    return incident_id, analysis, matches, round(time.time() - t0, 2)


def all_incidents():
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()]


def reset_db():
    import shutil
    if os.path.exists(DB):
        os.remove(DB)
    shutil.rmtree(PHOTO_DIR, ignore_errors=True)
    init_db()
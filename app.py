import json
import os

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="MtaaWatch AI", page_icon="🇺🇬", layout="wide")
core.init_db()


def show_photo(path, container):
    if path and os.path.exists(path):
        container.image(path, width=260)

CATEGORY_LABELS = {
    "blocked_drainage": "🌊 Blocked drainage / flooding",
    "uncollected_waste": "🗑️ Uncollected rubbish",
    "damaged_road": "🛣️ Damaged road",
    "water_point": "🚰 Water point problem",
    "power_hazard": "⚡ Electrical hazard (fallen pole / wires)",
    "other": "📋 Other issue",
}
URGENCY_LABELS = {
    "low": "🟢 Low",
    "medium": "🟡 Medium",
    "high": "🔴 High, needs quick attention",
    "unknown": "⚪ Not sure yet",
}

st.title("MtaaWatch AI 🇺🇬")
st.caption(
    "Community reports → structured cases → human review. "
    "AI flags are preliminary. Nothing here is sent to any authority."
)

tab_report, tab_review, tab_dash = st.tabs(["📝 Report an issue", "🔍 Review queue", "📊 Dashboard"])

# ------------------------------------------------------------------ report
with tab_report:
    with st.form("report_form", clear_on_submit=True):
        description = st.text_area(
            "What is the problem? (You can write in English, Luganda or Swahili)",
            placeholder="The drain near the market has been blocked for days and water is collecting on the road.",
        )
        location = st.text_input("Where? (village, market, road, landmark)")
        photo = st.file_uploader(
            "Add a photo (optional, but it helps). On a phone you can take one now.",
            type=["jpg", "jpeg", "png"],
        )
        st.caption("Please avoid photographing people's faces or vehicle number plates.")
        submitted = st.form_submit_button("Submit report")

    if submitted:
        if len(description.strip()) < 10:
            st.warning("Please describe the problem in a bit more detail.")
        else:
            with st.spinner("Analyzing report..."):
                iid, analysis, matches, secs = core.submit_report(
                    description.strip(), location, photo.getvalue() if photo else None
                )
            st.success(f"✅ Thank you! Your report has been received. Reference number: #{iid}")
            if photo and not analysis.get("photo_path"):
                st.warning("Your report was saved, but the photo could not be used. "
                           "Please make sure it is a JPG or PNG image under 8 MB.")
            with st.container(border=True):
                st.markdown("### What we understood")
                if analysis.get("photo_path"):
                    st.image(analysis["photo_path"], caption="Your photo", width=320)
                st.markdown(f"**Problem type:** {CATEGORY_LABELS[analysis['category']]}")
                st.markdown(f"**Location:** {analysis['location'] or 'Not provided'}")
                if analysis.get("language") not in (None, "unknown", "English"):
                    st.markdown(f"**Language detected:** {analysis['language']}")
                if analysis["source"] == "fallback":
                    st.warning("The AI could not be reached, so this report was NOT translated or "
                               "understood. A reviewer will have to read it manually.")
                st.markdown("**Summary (in English, for reviewers):**")
                st.text(analysis["summary"])
                st.markdown(f"**Seriousness (preliminary):** {URGENCY_LABELS[analysis['urgency']]}")
                st.markdown(f"**Who usually handles this:** {core.DEPARTMENTS[analysis['category']]}")
            st.info(
                "**What happens next:** A community reviewer will check your report. "
                "This is not yet sent to any authority, and the seriousness rating is only an automatic first guess."
            )
            if matches:
                st.warning(
                    "Similar reports about this problem already exist, so a reviewer "
                    "will check whether they are the same issue."
                )
            with st.expander("Technical details (for reviewers)"):
                st.caption(f"Processed in {secs}s · analysis by: {analysis['source']}")
                if analysis.get("error"):
                    st.caption(f"AI model error: {analysis['error']}")
                for score, old in matches:
                    st.text(f"Similar to #{old['id']} (score {score}): {old['summary']}")

# ------------------------------------------------------------------ review
with tab_review:
    incidents = core.all_incidents()
    by_id = {i["id"]: i for i in incidents}
    pending = [i for i in incidents if i["status"] == "Possible duplicate"]
    st.subheader(f"Possible duplicates awaiting a human decision ({len(pending)})")
    if not pending:
        st.info("No suspected duplicates right now.")
    for inc in pending:
        orig = by_id.get(inc["duplicate_of"])
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.markdown(f"**New report #{inc['id']}** · {inc['category']} · {inc['urgency']}")
            c1.text(inc["description"])
            if inc.get("source") == "gemma":
                c1.caption(f"English: {inc['summary']}")
            c1.caption(f"Location: {inc['location'] or 'not provided'}")
            show_photo(inc.get("photo_path"), c1)
            if orig:
                c2.markdown(f"**Existing case #{orig['id']}** · {orig['category']} · {orig['urgency']}")
                c2.text(orig["description"])
                if orig.get("source") == "gemma":
                    c2.caption(f"English: {orig['summary']}")
                c2.caption(f"Location: {orig['location'] or 'not provided'}")
                show_photo(orig.get("photo_path"), c2)
            st.caption(f"Similarity score: {inc['dup_score']}")
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Same incident", key=f"yes{inc['id']}"):
                core.confirm_duplicate(inc["id"])
                st.rerun()
            if b2.button("Different", key=f"no{inc['id']}"):
                core.reject_duplicate(inc["id"])
                st.rerun()

    st.subheader("Open cases")
    cases = [i for i in incidents if i["status"] not in ("Merged", "Rejected")]
    merged_count = {}
    for i in incidents:
        if i["status"] == "Merged" and i["duplicate_of"]:
            merged_count[i["duplicate_of"]] = merged_count.get(i["duplicate_of"], 0) + 1
    for inc in cases:
        n = merged_count.get(inc["id"], 0)
        with st.container(border=True):
            st.markdown(
                f"**#{inc['id']} · {inc['category']} · urgency: {inc['urgency']}** "
                f"· {n + 1} report(s) · status: {inc['status']}"
            )
            if inc.get("source") == "fallback":
                st.warning("The AI was unavailable, so this report was NOT translated or understood. "
                           "Needs human triage.")
                st.text(f"Original text: {inc['description']}")
            else:
                st.text(f"English summary: {inc['summary']}")
                if inc["description"].strip() != inc["summary"].strip():
                    st.caption(f"Original ({inc.get('language', 'unknown')}): {inc['description']}")
            if inc["status"] == "Possible duplicate" and inc.get("duplicate_of"):
                st.info(f"Possibly the same as case #{inc['duplicate_of']}. "
                        "Decide in the Review queue tab (nothing is merged automatically).")
            if inc.get("ai_error"):
                st.caption(f"AI error: {inc['ai_error']}")
            st.caption(f"Location: {inc['location'] or 'not provided'} · Suggested: {inc['department']}")
            photos = [inc.get("photo_path")] + [
                m.get("photo_path") for m in incidents
                if m["status"] == "Merged" and m["duplicate_of"] == inc["id"]
            ]
            photos = [p for p in photos if p and os.path.exists(p)]
            if photos:
                st.image(photos, width=160, caption=[f"Photo {k + 1}" for k in range(len(photos))])
            new_status = st.selectbox(
                "Status", ["Needs review", "Verified", "Escalated", "Resolved", "Rejected"],
                index=["Needs review", "Verified", "Escalated", "Resolved", "Rejected"].index(inc["status"])
                if inc["status"] in ["Needs review", "Verified", "Escalated", "Resolved", "Rejected"] else 0,
                key=f"st{inc['id']}",
            )
            if new_status != inc["status"]:
                core.set_status(inc["id"], new_status)
                st.rerun()

# --------------------------------------------------------------- dashboard
with tab_dash:
    incidents = core.all_incidents()
    if not incidents:
        st.info("No reports yet. Submit one, or load the demo data from the sidebar.")
    else:
        df = pd.DataFrame(incidents)
        merged = int((df["status"] == "Merged").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw reports", len(df))
        c2.metric("Organized cases", len(df) - merged)
        c3.metric("Duplicates merged by a human", merged)
        c4.metric("High urgency flags", int((df["urgency"] == "high").sum()))
        a, b = st.columns(2)
        a.subheader("By category")
        a.bar_chart(df["category"].value_counts())
        b.subheader("By status")
        b.bar_chart(df["status"].value_counts())
        st.subheader("By location")
        loc = df[df["location"].str.strip() != ""]["location"].str.title().value_counts()
        st.bar_chart(loc)
        st.subheader("All reports")
        st.dataframe(df[["id", "category", "location", "urgency", "status", "duplicate_of", "dup_score", "summary"]],
                     use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo tools")
    st.caption(f"Model: {core.MODEL} ({core.PROVIDER})")
    if core.PROVIDER == "gemini" and not core.API_KEY:
        st.error("GEMINI_API_KEY is not set. Reports will use the basic keyword fallback.")
    if st.button("Load 10 sample reports"):
        with open("sample_reports.json") as f:
            samples = json.load(f)
        with st.spinner("Processing samples..."):
            for s in samples:
                core.submit_report(s["description"], s["location"])
        st.rerun()
    if st.button("Reset database"):
        core.reset_db()
        st.rerun()
    st.caption("Prototype only. Flags are preliminary and need human verification.")
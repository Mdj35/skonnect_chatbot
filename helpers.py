import re
import random
import pandas as pd

# Category keywords used to map free-text interests to event categories
category_keywords = {
    "Education Support": ["education", "school", "scholarship", "study", "tuition", "exam", "learning"],
    "Environmental Protection": ["environment", "tree", "clean up", "recycle", "nature", "river", "eco"],
    "Health": ["health", "clinic", "hospital", "medical", "doctor", "checkup", "vaccine"],
    "Sports Development": ["sports", "basketball", "volleyball", "football", "athletics", "soccer", "games"],
    "Capability Building": ["training", "seminar", "workshop", "capacity", "skills", "orientation"],
    "General Administration": ["admin", "office", "barangay", "coordination", "support", "meeting"],
    "Youth Empowerment": ["youth", "leadership", "empowerment", "volunteer", "talent"]
}


def categorize_user_interest(interest_text):
    """Map a free-text interest to a known category using keyword matching.

    Returns the category string or None if no match.
    """
    if not interest_text:
        return None
    # reject inputs that are not meaningful sentences (e.g., single punctuation, emojis, urls)
    if not is_valid_text(interest_text):
        return None
    interest = interest_text.lower()
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", interest):
                return category
    # fallback: substring match
    for category, keywords in category_keywords.items():
        if any(kw in interest for kw in keywords):
            return category
    return None


def ensure_minimum_words(text, min_words=40):
    words = text.split()
    if len(words) < min_words:
        filler = (
            " To provide more context, our SK council continuously develops programs "
            "and projects to benefit the youth and community. We encourage participation "
            "and feedback so everyone in Brgy. Buhangin benefits from our initiatives."
        )
        text += filler
    return text


def is_valid_text(s: str, min_len: int = 2) -> bool:
    """Very small heuristic to detect whether a user-provided string is valid for categorization.

    Returns False for empty strings, strings with only punctuation, or obvious URLs.
    """
    if not s:
        return False
    s = s.strip()
    if len(s) < min_len:
        return False
    
    cleaned = re.sub(r"[\W_]+", "", s)
    if cleaned == "":
        return False
    # avoid URLs or emails
    if re.search(r"https?://|www\.|@\w+\.", s):
        return False
    # avoid strings made only of digits
    if re.fullmatch(r"\d+", cleaned):
        return False

    # simple gibberish heuristics:
    # 1) vowel ratio: valid English-like phrases usually contain vowels; low vowel ratio can indicate gibberish
    letters = re.sub(r"[^A-Za-z]", "", s)
    if letters:
        vowels = len(re.findall(r"[aeiouAEIOU]", letters))
        vowel_ratio = vowels / len(letters)
        # if almost no vowels (e.g., 'fddf') treat as invalid for short strings
        if vowel_ratio < 0.15 and len(letters) <= 12:
            return False
        # For longer single-word letter sequences, require a slightly higher vowel ratio
        if len(letters) > 12 and vowel_ratio < 0.20:
            return False

        # consonant-heavy sequences (very low vowel ratio) are likely gibberish
        consonant_ratio = (len(letters) - vowels) / len(letters)
        if consonant_ratio > 0.90 and len(letters) > 6:
            return False

        # if there's a single long uninterrupted letter run (no spaces) that's suspicious
        words = re.findall(r"[A-Za-z]+", s)
        for w in words:
            if len(w) > 30:
                return False

    # 2) repeated character runs (e.g., 'aaaaaa' or 'zzzz')
    if re.search(r"(.)\1{4,}", s):
        return False

    # 3) long random-like sequences of letters without vowels or structure
    # e.g., 'djfgdsjfgsdajkfgsdakeu' has very few vowels and many consonants; already covered
    # but add an extra check for words made of mostly consonants with length > 8
    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        v = len(re.findall(r"[aeiouAEIOU]", w))
        if len(w) >= 8 and (v / len(w) if len(w) else 0) < 0.18:
            return False

    return True


def handle_undetermined_input():
    """Return a friendly prompt when user input is not usable for classification.

    Keep the message short and guide the user to provide clearer input.
    """
    return (
        "I didn't quite get that. Could you type that as a short sentence or phrase? "
        "For example: 'I'm interested in youth training' or 'health clinic events'."
    )


def recommend_sample_faq_questions(faq_df, n=5):
    """Return up to `n` example user patterns from the faq dataframe.

    Prefers sampling distinct short patterns that look like user questions/phrases.
    """
    if faq_df is None or faq_df.empty:
        return []
    if "patterns" not in faq_df.columns:
        # try other plausible columns
        for col in ["user_message", "question", "pattern"]:
            if col in faq_df.columns:
                patterns = faq_df[col].astype(str).tolist()
                break
        else:
            return []
    else:
        patterns = faq_df["patterns"].astype(str).tolist()

    # clean and select short distinct patterns
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in patterns if p and len(p.strip()) > 2]
    seen = set()
    out = []
    for p in cleaned:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= n:
            break
    return out


# reply templates and state to avoid repeating identical replies
templates = [
    "Thanks — {answer}",
    "Here’s a concise answer: {answer}",
    "Happy to help! {answer}",
    "I found this for you: {answer}",
    "Quick info: {answer}",
    "Here you go — {answer}",
    "Good question — {answer}",
]
# short friendly sentence appended to many replies to close the turn and invite follow-up
enders = [
    "If that helps, tell me which one you want more details about.",
    "Would you like more details on any of these?",
    "Happy to show more info — just ask which one.",
    "You can ask me for details or other interests anytime.",
]
last_responses = {}


def format_faq_answer(answer: str, user_text: str | None = None) -> str:
    """Wrap FAQ answers in a short, friendly preface and polite ender."""
    base = str(answer or "").strip()
    if not base:
        return "Sorry, I don't have an answer for that right now. " + random.choice(enders)
    # Keep it short and actionable, add a soft ender
    preface = f"Sure — {base}"
    return f"{preface} {random.choice(enders)}"


def construct_reply_from_recommendations(events: list, category: str | None = None, limit: int | None = None) -> str:
    """Create a friendly, readable reply from a list of event recommendation dicts.

    Adds a polite ender and a short actionable suggestion. Designed to be user-facing.
    """
    if not events:
        return "Sorry — I couldn't find any events matching that. " + random.choice(enders)
    count = len(events) if limit is None else min(len(events), limit)
    cat_text = f" for {category}" if category else ""
    intro = f"I found {count} event(s){cat_text}:"
    summaries = [e.get("summary", "").strip() for e in events if e.get("summary")]
    body = "\n\n".join(summaries) if summaries else ", ".join([e.get("ppa", "") for e in events])
    follow_up = "If you'd like, ask me to show more details or filter by another interest."
    end = random.choice(enders)
    return f"{intro}\n\n{body}\n\n{follow_up} {end}"


def generate_dynamic_reply(base_reply, intent, user_text: str = None):
    """Create a friendly dynamic reply while avoiding repeated identical messages.

    - Uses short preface templates.
    - If base_reply looks like a direct FAQ answer, prefer format_faq_answer.
    - If base_reply contains event summaries (emoji or multi-line), keep it intact and add an ender.
    """
    base = str(base_reply or "").strip()

    # If the base looks like an event block (emoji summaries / multi-line), don't add extra preface but append ender
    looks_like_events = "📌" in base or ("\n" in base and len(base.splitlines()) > 1)

    if looks_like_events:
        reply = f"{base}\n\n{random.choice(enders)}"
    else:
        # If this appears to be an FAQ-style short answer, wrap it politely
        if len(base.split()) <= 30:
            reply = format_faq_answer(base, user_text=user_text)
        else:
            chosen_template = random.choice(templates)
            reply = chosen_template.format(answer=base) + " " + random.choice(enders)

    # avoid repeating identical last response for same intent
    if intent in last_responses and last_responses[intent] == reply:
        alt_templates = [t for t in templates if t.format(answer=base) != reply]
        if alt_templates:
            reply = random.choice(alt_templates).format(answer=base) + " " + random.choice(enders)

    # ensure a reasonable minimum length but add a short, neutral filler only when necessary
    reply = ensure_minimum_words(reply, 20)
    last_responses[intent] = reply
    return reply


def clean_string(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Reference Code" in df.columns:
        df = df.drop_duplicates(subset=["Reference Code"])
    if {"PPAs", "Description"}.issubset(df.columns):
        df = df.copy()
        df["fingerprint"] = (
            df["PPAs"].astype(str).str.lower().str.strip() + "||" +
            df["Description"].astype(str).str.lower().str.strip()
        )
        df = df.drop_duplicates(subset=["fingerprint"]).drop(columns=["fingerprint"])
    return df


def make_summary_from_row(row):
    return (
        f"📌 {clean_string(row.get('PPAs',''))} ({clean_string(row.get('Recommended Category',''))})\n"
        f"📝 {clean_string(row.get('Description',''))}\n"
        f"🎯 Expected Result: {clean_string(row.get('Expected Result',''))}\n"
        f"📅 Implementation: {clean_string(row.get('Period of Implementation',''))}\n"
        f"👥 Responsible: {clean_string(row.get('Person Responsible',''))}\n"
        f"🔖 Reference: {clean_string(row.get('Reference Code',''))}"
    )


def recommend_event_all(event_df: pd.DataFrame, category: str | None = None, limit: int | None = None):
    """
    If category is provided (string), return events matching that category.
    If category is None or empty, return general/top events across all categories.
    """
    if event_df is None or event_df.empty:
        return []

    # normalize category parameter
    if category is not None:
        category = str(category).strip()
        if category == "":
            category = None

    # when category provided, filter by normalized category
    if category:
        mask = event_df["Recommended Category"].astype(str).str.lower().str.strip() == category.lower()
        matches = event_df[mask].copy()
    else:
        # no category -> consider all events, maybe prioritize by a column if available
        matches = event_df.copy()

    if matches.empty:
        return []

    # dedupe and optionally sort (if you have e.g. 'Period of Implementation' or a 'priority' column)
    matches = dedupe_events(matches)

    # If there's a timestamp/priority column you can sort here; otherwise keep original order
    # Example (uncomment if you have a date column): matches = matches.sort_values("Start Date", ascending=True)

    if limit and len(matches) > limit:
        matches = matches.head(limit)

    return [{
        "reference_code": clean_string(row.get("Reference Code", "")),
        "ppa": clean_string(row.get("PPAs", "")),
        "description": clean_string(row.get("Description", "")),
        "expected_result": clean_string(row.get("Expected Result", "")),
        "period": clean_string(row.get("Period of Implementation", "")),
        "responsible": clean_string(row.get("Person Responsible", "")),
        "category": clean_string(row.get("Recommended Category", "")),
        "summary": make_summary_from_row(row),
    } for _, row in matches.iterrows()]



def find_faq_answer(user_text: str, faq_df: pd.DataFrame):
    """Try to find a matching FAQ answer for the user's text.

    Matching rules (simple, deterministic):
    - Look for a 'patterns' / 'pattern' / 'user_message' / 'question' column.
    - Split multi-pattern cells on common separators and try exact or substring matches.
    - If a row matches, return its 'answer' / 'response' / 'reply' column if present, otherwise
      return the matched pattern text.
    """
    if faq_df is None or faq_df.empty or not user_text:
        return None

    user = user_text.strip().lower()
    candidate_cols = [c for c in ["patterns", "pattern", "user_message", "question"] if c in faq_df.columns]
    if not candidate_cols:
        return None

    answer_cols = [c for c in ["answer", "response", "reply", "answers"] if c in faq_df.columns]

    splitter = re.compile(r"[|\n;]+|,\s+")
    for _, row in faq_df.iterrows():
        for col in candidate_cols:
            cell = str(row.get(col, "") or "")
            if not cell:
                continue
            parts = [p.strip() for p in splitter.split(cell) if p and len(p) > 1]
            # include whole cell as fallback
            if not parts:
                parts = [cell.strip()]
            for p in parts:
                lp = p.lower()
                # exact match or substring either way (favor exact)
                if lp == user or lp in user or user in lp:
                    if answer_cols:
                        for ac in answer_cols:
                            ans = row.get(ac)
                            if ans and str(ans).strip():
                                return str(ans).strip()
                    # fallback to returning the matched pattern text
                    return p
    return None


def recommend_for_user_input(user_text: str, faq_df: pd.DataFrame = None, event_df: pd.DataFrame = None, limit: int = 3):
    """High-level helper: prefer FAQ answers when available, otherwise fall back to event recommendations.

    Returns a dict with:
      - type: 'faq' | 'events' | 'none'
      - reply: formatted string ready for user
      - recommendations: list (for events) or []
      - category: category string when applicable
    """
    # 1) Try exact/substring FAQ match
    faq_answer = find_faq_answer(user_text, faq_df)
    if faq_answer:
        reply = format_faq_answer(faq_answer, user_text=user_text)
        return {"type": "faq", "reply": reply, "recommendations": [], "category": None}

    # 2) If no FAQ match, try to categorize interest and return events
    cat = categorize_user_interest(user_text)
    if cat and event_df is not None:
        events = recommend_event_all(event_df, cat, limit=limit)
        if events:
            reply = construct_reply_from_recommendations(events, category=cat, limit=limit)
            return {"type": "events", "reply": reply, "recommendations": events, "category": cat}

    # 3) Nothing matched
    msg = handle_undetermined_input()
    # attach a gentle ender to keep tone consistent
    reply = f"{msg} {random.choice(enders)}"
    return {"type": "none", "reply": reply, "recommendations": [], "category": None}

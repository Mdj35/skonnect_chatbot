from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, datetime, random, re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from difflib import SequenceMatcher
from helpers import (
    categorize_user_interest,
    generate_dynamic_reply,
    ensure_minimum_words,
    clean_string,
    dedupe_events,
    make_summary_from_row,
    recommend_event_all,
    is_valid_text,
    handle_undetermined_input,
    recommend_sample_faq_questions,
    find_faq_answer,
    format_faq_answer,
)
from db_connection import fetch_events_from_mysql

user_last_intent = {}
user_last_topic = {}

conversation_memory = {}

def remember(user_id, message, intent, response):
    """Keep a short per-user conversation buffer (max 10 entries)."""
    try:
        uid = user_id or "guest"
        if uid not in conversation_memory:
            conversation_memory[uid] = []
        conversation_memory[uid].append({
            "timestamp": datetime.datetime.now(),
            "message": message,
            "intent": intent,
            "response": response
        })
        if len(conversation_memory[uid]) > 10:
            conversation_memory[uid].pop(0)
    except Exception:
        # non-fatal if memory update fails
        pass

# remove the big model / tokenizer / data initialization here and import them instead
from model_loader import (
    response_generator,
    faq_model,
    event_model,
    faq_tokenizer,
    faq_label_encoder,
    event_tokenizer,
    event_label_encoder,
    faq_df,
    event_df,
    faq_max_len,
    event_max_len,
    vectorizer,
    clf,
    LOG_FILE,
)

def merge_event_sources(csv_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    """Combine MySQL and CSV events into one unified dataset."""
    if csv_df is None or csv_df.empty:
        return db_df
    if db_df is None or db_df.empty:
        return csv_df
    
    db_df = db_df.rename(columns={
        "title": "PPAs",
        "description": "Description",
        "event_type": "Recommended Category",
        "id": "Reference Code",
    })
    
    # Normalize columns and merge
    common_cols = ["Reference Code", "PPAs", "Description", "Recommended Category"]
    merged = pd.concat([csv_df[common_cols], db_df[common_cols]], ignore_index=True).fillna("")
    return merged

faq_df = pd.read_csv("skonnect_faq_dataset_intents.csv")

# Normalize FAQ text columns for case-insensitive matching
faq_df["intent"] = faq_df["intent"].astype(str).str.strip()
faq_df["patterns"] = faq_df["patterns"].astype(str).str.strip().str.lower()
faq_df["bot_response"] = faq_df["bot_response"].astype(str)

# Optional: normalize other columns if needed
faq_df.columns = faq_df.columns.str.strip()

# Load both CSV and DB events
csv_events = pd.read_csv("Events.csv").fillna("")
db_events = fetch_events_from_mysql()
event_df = merge_event_sources(csv_events, db_events)


# Normalize text columns
for col in ["Recommended Category", "PPAs", "Description", "Reference Code"]:
    if col in event_df.columns:
        event_df[col] = event_df[col].astype(str)

faq_max_len, event_max_len = 20, 50

vectorizer = HashingVectorizer(n_features=2**16)
clf = SGDClassifier(loss="log_loss")

if {"intent", "patterns"}.issubset(faq_df.columns):
    X_init = vectorizer.transform(faq_df["patterns"].astype(str).tolist())
    y_init = faq_df["intent"].astype(str).tolist()
    try:
        clf.partial_fit(X_init, y_init, classes=np.unique(y_init))
    except Exception:
        pass
        
LOG_FILE = "chat_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "user_id", "username", "user_message", "predicted_intent",
        "bot_response", "model_source"
    ]).to_csv(LOG_FILE, index=False)


def format_db_events_for_reply(db_events_df: pd.DataFrame, limit=None) -> str:
    """Format MySQL events into readable text with month names (no weekday) and no time/duration."""
    if db_events_df is None or db_events_df.empty:
        return ""
    
    if limit:
        db_events_df = db_events_df.head(limit).copy()
    else:
        db_events_df = db_events_df.copy()

    # parse date robustly (try day-first then month-first)
    def parse_to_datetime(val):
        if pd.isna(val) or val == "":
            return pd.NaT
        try:
            dt = pd.to_datetime(val, infer_datetime_format=True, dayfirst=True, errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(val, infer_datetime_format=True, dayfirst=False, errors='coerce')
            return dt
        except Exception:
            return pd.NaT

    db_events_df["__parsed_date"] = db_events_df.get("date", pd.Series([""] * len(db_events_df))).apply(parse_to_datetime)

    lines = []
    for _, row in db_events_df.iterrows():
        title = str(row.get("title", "")).strip()
        desc = str(row.get("description", "")).strip()
        loc = str(row.get("location", "")).strip()
        etype = str(row.get("event_type", "")).strip()

        # Date formatting (Month day, Year) without weekday or time/duration
        parsed_date = row.get("__parsed_date")
        if pd.notna(parsed_date):
            date_str = f"{parsed_date.strftime('%B')} {int(parsed_date.day)}, {int(parsed_date.year)}"
        else:
            raw_date = str(row.get("date", "")).strip()
            # try parsing raw_date to extract a real date
            try_dt = pd.to_datetime(raw_date, infer_datetime_format=True, errors='coerce')
            if not pd.isna(try_dt):
                date_str = f"{try_dt.strftime('%B')} {int(try_dt.day)}, {int(try_dt.year)}"
            else:
                # remove things like "0 days 08:00:00" and standalone time strings
                cleaned = re.sub(r'\d+\s+days\s+\d{1,2}:\d{2}:\d{2}', '', raw_date)
                cleaned = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', cleaned).strip()
                date_str = cleaned

        # Do NOT include time — only the date string
        datetime_line = date_str or ""

        formatted = f"📌 {title} ({etype})\n📝 {desc}\n📅 {datetime_line}\n📍 {loc}"
        lines.append(formatted)
    
    return "\n\n".join(lines)

def log_conversation(user_message, predicted_intent, bot_reply, source="keras", user_id=None, username=None):
    """Append a new conversation entry to the logs, update short-term memory and retrain the incremental model."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([[timestamp, user_id or "", username or "", user_message, predicted_intent, bot_reply, source]],
                          columns=["timestamp", "user_id", "username", "user_message", "predicted_intent",
                                   "bot_response", "model_source"])
    try:
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    except Exception as e:
        print("Log write failed:", e)

    try:
        remember(user_id or "guest", user_message, predicted_intent, bot_reply)
    except Exception:
        pass

    if user_id:
        user_last_intent[user_id] = predicted_intent

        if predicted_intent in ["faq_direct_match", "office_hours", "programs_joining"]:
            # Try to extract main keyword/topic from the user message
            topic_match = re.findall(
    r"\b("
    r"role|role of sk|sk council|current officials|officials|office location|office located|"
    r"office hours|contact|how to contact|announcements|upcoming activities|activities|"
    r"katipunan ng kabataan|kk|kk members|how old|age|register|registration|documents|"
    r"out-of-school|out of school|working students|jobs|projects|programs|sports|"
    r"basketball|volleyball|scholarship|educational|livelihood|skills training|health|"
    r"wellness|environment|tree planting|join|how to join|registration fees|fees"
    r")\b",
    user_message.lower()
)

            if topic_match:
                user_last_topic[user_id] = topic_match[-1]

    # incremental retraining from logs (best-effort)
    try:
        log_data = pd.read_csv(LOG_FILE)
        if not log_data.empty:
            X_logs = vectorizer.transform(log_data["user_message"].astype(str).tolist())
            y_logs = log_data["predicted_intent"].astype(str).tolist()
            if len(set(y_logs)) > 0:
                clf.partial_fit(X_logs, y_logs, classes=np.unique(y_logs))
    except Exception as e:
        print("Log reload skipped:", e)


def get_recent_context(user_id, window=3):
    """Fetch recent user messages for conversation continuity."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        logs = pd.read_csv(LOG_FILE)
        logs = logs[logs["user_id"] == user_id]
        if logs.empty:
            return []
        return logs.tail(window)["user_message"].astype(str).tolist()
    except Exception as e:
        print("Context load failed:", e)
        return []


def classify_message(message, event_threshold=0.70, faq_threshold=0.40):
    """Classify message into FAQ or Event domain with confidence thresholds."""
    # FAQ prediction
    faq_seq = faq_tokenizer.texts_to_sequences([message])
    faq_padded = pad_sequences(faq_seq, maxlen=faq_max_len, padding="post")
    faq_pred = faq_model.predict(faq_padded)
    faq_conf = float(np.max(faq_pred))
    faq_intent = faq_label_encoder.inverse_transform([np.argmax(faq_pred)])[0]

    # Event prediction
    event_seq = event_tokenizer.texts_to_sequences([message])
    event_padded = pad_sequences(event_seq, maxlen=event_max_len, padding="post")
    event_pred = event_model.predict(event_padded)
    event_conf = float(np.max(event_pred))
    event_intent = event_label_encoder.inverse_transform([np.argmax(event_pred)])[0]

    # Check for explicit event keywords
    msg = message.lower()
    explicit_event_request = bool(
        re.search(r"\b(show|list|find|display|give me|all|upcoming|what events|when are|show me)\b", msg)
        and re.search(r"\bevents?\b", msg)
    )

    if explicit_event_request:
        if event_conf >= event_threshold:
            return "event", event_intent, event_conf
        return "faq", faq_intent, faq_conf

    if faq_conf >= faq_threshold:
        return "faq", faq_intent, faq_conf
    if event_conf >= event_threshold:
        return "event", event_intent, event_conf
    return "faq", faq_intent, faq_conf

def get_last_intent_and_topic(user_id):
    """Retrieve the last recorded intent and topic for a user from logs."""
    if not os.path.exists(LOG_FILE):
        return None, None
    try:
        logs = pd.read_csv(LOG_FILE)
        user_logs = logs[logs["user_id"] == user_id]
        if user_logs.empty:
            return None, None
        last_row = user_logs.iloc[-1]
        last_intent = str(last_row["predicted_intent"])
        last_message = str(last_row["user_message"]).lower()

        # detect topic from message text
        topic_match = re.findall(r"\b(sk council|office hours|program|event|fees?|activities?)\b", last_message)
        last_topic = topic_match[-1] if topic_match else last_message
        return last_intent, last_topic
    except Exception as e:
        print("Failed to get last intent/topic:", e)
        return None, None

def find_closest_faq_match(user_text: str, faq_df: pd.DataFrame, threshold: float = 0.50):
    """
    Detects if user's message closely matches any FAQ pattern or intent, 
    even with typos or paraphrases.
    Returns the best match if above threshold.
    """
    if not user_text or faq_df is None or faq_df.empty:
        return None

    user_clean = re.sub(r"[^\w\s]", "", user_text.lower().strip())
    best_match = None
    best_score = 0.0

    for _, row in faq_df.iterrows():
        for col in ["patterns", "intent"]:
            # skip missing values safely
            if col not in row or pd.isna(row[col]):
                continue
            clean_pattern = re.sub(r"[^\w\s]", "", str(row[col]).lower().strip())
            score = SequenceMatcher(None, user_clean, clean_pattern).ratio()
            if score > best_score:
                best_score = score
                best_match = row

    return best_match if best_score >= threshold else None


def handle_low_confidence_intent(message, predicted_intent, confidence, faq_df, username, threshold=0.4):
    """
    Handle low-confidence FAQ or Event intent predictions gracefully.
    Suggest likely related FAQ questions instead of returning 'low threshold'.
    Provides conversational suggestions and optional neuralized phrasing.
    """
    if confidence >= threshold:
        return None

    # Try to find close FAQ patterns (fuzzy)
    user_clean = re.sub(r"[^\w\s]", "", message.lower().strip())
    faq_copy = faq_df.copy()
    faq_copy["clean_patterns"] = faq_copy["patterns"].astype(str).apply(lambda x: re.sub(r"[^\w\s]", "", x.lower().strip()))
    faq_copy["similarity"] = faq_copy["clean_patterns"].apply(lambda x: SequenceMatcher(None, user_clean, x).ratio())
    similar_faqs = faq_copy.sort_values("similarity", ascending=False).head(6)

    suggestions = []
    for _, r in similar_faqs.iterrows():
        if pd.isna(r["patterns"]):
            continue
        suggestions.append(r["patterns"])

    # fallback to random sample if no good similar found
    if not suggestions:
        try:
            sample_qs = faq_df["patterns"].dropna().tolist()
            suggestions = random.sample(sample_qs, min(3, len(sample_qs)))
        except Exception:
            suggestions = []

    # Build conversational reply
    if similar_faqs.iloc[0:1]["similarity"].values and float(similar_faqs.iloc[0:1]["similarity"].values[0]) >= 0.45:
        bot_reply_base = "I’m not completely sure, but you might be asking something like:\n" + "\n".join(f"- {q}" for q in suggestions[:3])
    else:
        bot_reply_base = "I’m not confident about that question. Try rephrasing, or you can ask one of these:\n" + "\n".join(f"- {q}" for q in suggestions[:3])

    # personal greeting
    if username and username.lower() not in ("guest", "none"):
        bot_reply_base = f"Hi {username.split()[0]}, " + bot_reply_base

    # Optionally include a neuralized suggestion (short)
    bot_reply = bot_reply_base
    try:
        # keep neural generation short — use first suggestion as seed
        if response_generator and suggestions:
            short_seed = suggestions[0]
            bot_reply = generate_neural_reply(bot_reply_base, short_seed, max_length=80)
    except Exception:
        pass

    return {
        "intent": "low_confidence_suggestion",
        "confidence": float(confidence),
        "response": bot_reply,
        "suggested_questions": suggestions[:3],
        "source": "confidence_handler"
    }

def generate_neural_reply(faq_answer: str, user_input: str, max_length: int = 150):
    """Optionally generate a more natural reply using a neural generator. Falls back to faq_answer."""
    try:
        if response_generator is None:
            return faq_answer
        prompt = f"User: {user_input}\nAnswer (use the information below): {faq_answer}\nRespond naturally:"
        out = response_generator(prompt, max_length=max_length, do_sample=True, temperature=0.7)
        return out[0].get("generated_text", faq_answer)
    except Exception:
        return faq_answer
# ...existing code...
def handle_low_confidence_intent(message, predicted_intent, confidence, faq_df, username, threshold=0.4):
    """
    Handle low-confidence FAQ or Event intent predictions gracefully.
    Suggest likely related FAQ questions instead of returning 'low threshold'.
    Provides conversational suggestions and optional neuralized phrasing.
    """
    if confidence >= threshold:
        return None

    # Try to find close FAQ patterns (fuzzy)
    user_clean = re.sub(r"[^\w\s]", "", message.lower().strip())
    faq_copy = faq_df.copy()
    faq_copy["clean_patterns"] = faq_copy["patterns"].astype(str).apply(lambda x: re.sub(r"[^\w\s]", "", x.lower().strip()))
    faq_copy["similarity"] = faq_copy["clean_patterns"].apply(lambda x: SequenceMatcher(None, user_clean, x).ratio())
    similar_faqs = faq_copy.sort_values("similarity", ascending=False).head(6)

    suggestions = []
    for _, r in similar_faqs.iterrows():
        if pd.isna(r["patterns"]):
            continue
        suggestions.append(r["patterns"])

    # fallback to random sample if no good similar found
    if not suggestions:
        try:
            sample_qs = faq_df["patterns"].dropna().tolist()
            suggestions = random.sample(sample_qs, min(3, len(sample_qs)))
        except Exception:
            suggestions = []

    # Build conversational reply
    if similar_faqs.iloc[0:1]["similarity"].values and float(similar_faqs.iloc[0:1]["similarity"].values[0]) >= 0.45:
        bot_reply_base = "I’m not completely sure, but you might be asking something like:\n" + "\n".join(f"- {q}" for q in suggestions[:3])
    else:
        bot_reply_base = "I’m not confident about that question. Try rephrasing, or you can ask one of these:\n" + "\n".join(f"- {q}" for q in suggestions[:3])

    # personal greeting
    if username and username.lower() not in ("guest", "none"):
        bot_reply_base = f"Hi {username.split()[0]}, " + bot_reply_base

    # Optionally include a neuralized suggestion (short)
    bot_reply = bot_reply_base
    try:
        # keep neural generation short — use first suggestion as seed
        if response_generator and suggestions:
            short_seed = suggestions[0]
            bot_reply = generate_neural_reply(bot_reply_base, short_seed, max_length=80)
    except Exception:
        pass

    return {
        "intent": "low_confidence_suggestion",
        "confidence": float(confidence),
        "response": bot_reply,
        "suggested_questions": suggestions[:3],
        "source": "confidence_handler"
    }

app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = (data.get("message") or "").strip()
    requested_limit = data.get("limit")
    raw_interest = data.get("interest")
    raw_interests = data.get("interests")

    user_id = data.get("user_id") or "guest"
    username = data.get("username") or user_id

    def personalize(text, username_val):
        """Adds the user's first name to the bot's response."""
        if not username_val:
            return text
        uname = str(username_val).strip()
        if not uname or uname.lower() in ("guest", "none"):
            return text
        short = uname.split()[0].strip()
        return f"Hi {short}, {text}"

    # Helper: neuralize, personalize and log a reply (used across branches)
    def finalize_reply(raw_text, intent_label, source_label):
        """
        Optionally runs the raw_text through the neural rephraser (if available),
        then personalizes and logs the final reply. Returns the final text.
        """
        try:
            neural = generate_neural_reply(raw_text, message, max_length=200) if response_generator else raw_text
        except Exception:
            neural = raw_text
        final_text = personalize(neural, username)
        # log_conversation will also update short-term memory
        try:
            log_conversation(message, intent_label, final_text, source_label, user_id, username)
        except Exception:
            pass
        return final_text

    recent_context = get_recent_context(user_id)

    if len(message.split()) < 5 and recent_context:
        message_for_classification = f"{' '.join(recent_context[-2:])}. {message.lower()}"
    else:
        message_for_classification = message.lower()

    # categorize interests
    categorized_interests = []
    if raw_interests and isinstance(raw_interests, list):
        for intr in raw_interests:
            cat = categorize_user_interest(intr)
            if cat and cat not in categorized_interests:
                categorized_interests.append(cat)
    elif raw_interest:
        cat = categorize_user_interest(raw_interest)
        if cat:
            categorized_interests.append(cat)

    # validation
    if not is_valid_text(message):
        suggestions = recommend_sample_faq_questions(faq_df, n=5)
        if suggestions:
            bot_reply = personalize(
                "I couldn't understand that. Try asking me one of these:\n" +
                "\n".join(f"- {q}" for q in suggestions),
                username
            )
            log_conversation(message, "undetermined", bot_reply, "validation", user_id, username)
            return jsonify({
                "intent": None,
                "confidence": 0.0,
                "response": bot_reply,
                "suggested_questions": suggestions,
                "source": "validation",
                "categorized_interests": []
            })
        prompt = handle_undetermined_input()
        bot_reply = personalize(prompt, username)
        log_conversation(message, "undetermined", bot_reply, "validation", user_id, username)
        return jsonify({
            "intent": None,
            "confidence": 0.0,
            "response": bot_reply,
            "source": "validation",
            "categorized_interests": []
        })

    # follow-up handler (yes / okay)
    yes_patterns = ["yes", "yeah", "yep", "sure", "okay", "ok", "yup", "please do", "go ahead"]
    try:
        if any(phrase in message.lower() for phrase in yes_patterns):
            last_intent = user_last_intent.get(user_id)
            last_topic = user_last_topic.get(user_id, "")

            if not last_intent or not last_topic:
                db_intent, db_topic = get_last_intent_and_topic(user_id)
                if db_intent:
                    last_intent = db_intent
                if db_topic:
                    last_topic = db_topic

            if not last_topic:
                try:
                    logs = pd.read_csv(LOG_FILE)
                    user_logs = logs[logs["user_id"] == user_id]
                    if not user_logs.empty:
                        last_intent = user_logs.iloc[-1]["predicted_intent"]
                        last_topic = str(user_logs.iloc[-1]["user_message"]).lower()
                except Exception:
                    pass

            topic_to_intent = {
                "role": "role_of_sk_council",
                "role of sk": "role_of_sk_council",
                "sk council role": "role_of_sk_council",
                "current officials": "current_sk_officials",
                "sk officials": "current_sk_officials",
                "office location": "sk_office_location",
                "office located": "sk_office_location",
                "office hours": "office_hours",
                "contact": "sk_contact",
                "contact person": "sk_contact",
                "how to contact": "sk_contact",
                "upcoming activities": "sk_activities",
                "announcements": "sk_announcements",
                "katipunan ng kabataan": "kk_membership",
                "kk members": "kk_membership",
                "how old": "kk_age_requirement",
                "age": "kk_age_requirement",
                "register": "kk_registration",
                "how to register": "kk_registration",
                "documents": "kk_documents",
                "what documents": "kk_documents",
                "out-of-school": "oosy_participation",
                "out of school": "oosy_participation",
                "working students": "working_students_participation",
                "jobs": "working_students_participation",
                "projects": "sk_projects",
                "programs": "sk_programs",
                "sports": "sk_sports",
                "basketball": "sk_sports",
                "volleyball": "sk_sports",
                "scholarship": "sk_educational_programs",
                "educational": "sk_educational_programs",
                "livelihood": "sk_livelihood_programs",
                "skills training": "sk_livelihood_programs",
                "health": "sk_health_projects",
                "wellness": "sk_health_projects",
                "environment": "sk_environmental_programs",
                "tree planting": "sk_environmental_programs",
                "join": "how_to_join",
                "registration fees": "activity_fees",
                "fees": "activity_fees",
            }

            def resolve_intent_from_topic(topic_text):
                if not topic_text:
                    return None
                txt = str(topic_text).lower()
                if txt in faq_df["intent"].astype(str).str.lower().tolist():
                    return txt
                for key, intent in topic_to_intent.items():
                    if key in txt:
                        return intent
                for possible in faq_df["intent"].astype(str).tolist():
                    if all(w in txt for w in str(possible).lower().split()):
                        return possible
                return None

            resolved_intent = resolve_intent_from_topic(last_topic) or resolve_intent_from_topic(last_intent)

            if resolved_intent:
                matched = faq_df[faq_df["intent"].astype(str).str.lower().str.contains(str(resolved_intent).lower(), na=False)]
                if not matched.empty:
                    actual_answer = matched["bot_response"].iloc[0] if "bot_response" in matched.columns else matched.iloc[0].to_dict()
                    bot_reply = personalize(format_faq_answer(actual_answer, user_text=message), username)
                    log_conversation(message, resolved_intent, bot_reply, "faq", user_id, username)
                    return jsonify({
                        "intent": resolved_intent,
                        "confidence": 1.0,
                        "response": bot_reply,
                        "source": "faq"
                    })

            direct_answer = find_faq_answer(last_topic or last_intent or message, faq_df)
            if direct_answer:
                bot_reply = personalize(format_faq_answer(direct_answer, user_text=message), username)
                log_conversation(message, "faq_direct_match", bot_reply, "faq", user_id, username)
                return jsonify({
                    "intent": "faq_direct_match",
                    "confidence": 1.0,
                    "response": bot_reply,
                    "source": "faq"
                })

            bot_reply = personalize(
                "Sure! What would you like to see next — upcoming events, announcements, SK programs, or details about KK membership?",
                username
            )
            log_conversation(message, "followup_generic", bot_reply, "faq", user_id, username)
            return jsonify({
                "intent": "followup_generic",
                "confidence": 1.0,
                "response": bot_reply,
                "source": "faq"
            })
    except Exception as e:
        print("Follow-up handling failed:", e)

    # -----------------------
    # Handle greetings (hi, hello, hey, good morning/afternoon/evening)
    # -----------------------
    greeting_patterns = re.compile(
        r"\b(hi|hello|hey|hiya|good morning|good afternoon|good evening)\b",
        flags=re.I,
    )

    # If the message is a short/pure greeting -> respond immediately
    if message and greeting_patterns.search(message) and len(message.split()) <= 3:
        base_reply = "I’m Skonnect Chatbot — how can I help you today?"
        reply = personalize(base_reply, username)
        log_conversation(message, "greeting", reply, "meta", user_id, username)
        return jsonify({
            "intent": "greeting",
            "confidence": 1.0,
            "response": reply,
            "source": "meta"
        })

    # If greeting appears but message is longer, strip greeting words and continue processing
    if message and greeting_patterns.search(message) and len(message.split()) > 3:
        message = greeting_patterns.sub("", message).strip()

    # ---------------------------
    # Clean system or chatbot mentions before classification
    # ---------------------------
    # remove explicit mentions like "Skonnect Chatbot", "hi Skonnect", "hello Skonnect"
    message = re.sub(
        r"\b(skonnect\s*chatbot|hi\s*skonnect|hello\s*skonnect)\b[.,!]*",
        "",
        message,
        flags=re.I,
    ).strip()

    # -----------------------
    # Handle polite/closing messages (thanks, thank you, ty, thx)
    # -----------------------
    thank_patterns = re.compile(r"\b(thanks|thank you|thank u|ty|thx|thanks a lot|thanks!)\b", flags=re.I)
    if message and thank_patterns.search(message.strip()):
        reply = "You’re welcome! If you need anything else, I’m here to help."
        reply = personalize(reply, username)
        log_conversation(message, "thanks", reply, "meta", user_id, username)
        return jsonify({
            "intent": "thanks",
            "confidence": 1.0,
            "response": reply,
            "source": "meta"
        })

    if re.search(r"\bgeneral events?\b", message.lower()):
        try:
            # Filter for general category (case-insensitive)
            general_events_df = event_df[event_df["Recommended Category"].astype(str).str.lower() == "general administration"]

            if general_events_df is None or general_events_df.empty:
                raw_reply = "There are currently no general events available right now. Please check back later!"
                bot_reply = finalize_reply(raw_reply, "general_events_empty", "event")
                return jsonify({
                    "intent": "general_events",
                    "confidence": 1.0,
                    "response": bot_reply,
                    "source": "event"
                })

            # Format events safely
            try:
                formatted_events = format_db_events_for_reply(general_events_df)
            except Exception as inner_e:
                print("⚠️ format_db_events_for_reply failed:", inner_e)
                formatted_events = "\n".join(
                    f"📌 {row.get('PPAs','').strip()} — {row.get('Description','').strip()}"
                    for _, row in general_events_df.iterrows()
                )

            # Ensure it's not empty
            if not formatted_events.strip():
                formatted_events = "\n".join(
                    f"📌 {row.get('PPAs','').strip()} — {row.get('Description','').strip()}"
                    for _, row in general_events_df.iterrows()
                )

            raw_reply = f"Here are the current General Events under 'General Administration':\n\n{formatted_events}"
            bot_reply = finalize_reply(raw_reply, "general_events", "event")

            return jsonify({
                "intent": "general_events",
                "confidence": 1.0,
                "response": bot_reply,
                "source": "event"
            })
        except Exception as e:
            print("🔥 General events handler crashed:", e)
            raw_reply = "Sorry, I ran into an issue while fetching general events. Please try again shortly."
            bot_reply = finalize_reply(raw_reply, "general_events_error", "event_error")
            return jsonify({
                "intent": "general_events",
                "confidence": 0.0,
                "response": bot_reply,
                "source": "event_error"
            })

    if not message.endswith("?") and any(w in message.lower() for w in ["what", "who", "where", "when", "how", "why", "can", "does", "is", "are"]):
        message = message + "?"

    possible_match = find_closest_faq_match(message, faq_df)
    if possible_match is not None:
        raw_answer = format_faq_answer(possible_match["bot_response"], user_text=message)
        bot_reply = finalize_reply(raw_answer, "faq_typo_match", "faq")
        return jsonify({
            "intent": possible_match["intent"],
            "confidence": 1.0,
            "response": bot_reply,
            "source": "faq_typo_match"
        })

    # classification
    model_type, predicted_intent, confidence = classify_message(message_for_classification)

    # Handle low-confidence predictions gracefully (for both FAQ and Event)
    low_conf_result = handle_low_confidence_intent(message, predicted_intent, confidence, faq_df, username)
    if low_conf_result:
        log_conversation(message, predicted_intent, low_conf_result["response"], low_conf_result["source"], user_id, username)
        return jsonify(low_conf_result)

    # event override if message explicitly mentions events
    event_trigger_phrases = [
        r"\b(show|list|display|give me|find|see|tell me about|what events|upcoming events?)\b"
    ]
    event_keywords = ["event", "events", "program", "programs", "activity", "activities"]

    # only override if the user explicitly asks to *see* or *list* events
    if model_type == "faq":
        msg_lower = message.lower()
        if any(re.search(p, msg_lower) for p in event_trigger_phrases) and any(kw in msg_lower for kw in event_keywords):
            model_type = "event"
            predicted_intent = "general_event_recommendation"

    # FAQ branch with whitelist
    if model_type == "faq":
        allowed_questions = [
            "what is the role of the sk council in barangay buhangin",
            "who are the current sk officials in brgy buhangin",
            "where is the sk office located in the barangay",
            "what are the office hours of the sk council",
            "how can i contact the sk chairperson or sk officials",
            "how do i know about upcoming sk activities and announcements",
            "who are considered members of the katipunan ng kabataan in brgy buhangin",
            "how old do i have to be to be part of the katipunan ng kabataan",
            "how can i register as a member of the kk",
            "what documents are needed to join or update my kk membership",
            "can out-of-school youth join sk activities",
            "can working students or youth with jobs still join sk programs",
            "are sk projects only for youth or can the whole community benefit",
            "what types of programs does the sk council usually implement",
            "does the sk provide sports activities like basketball or volleyball tournaments",
            "does the sk have educational programs like scholarships or tutorials",
            "are there livelihood or skills training programs for the youth",
            "what health and wellness projects are available for the youth",
            "does the sk support environmental programs like tree planting or clean-up drives",
            "how do i join sk programs or activities",
            "are there registration fees for sk activities"
        ]
        # compare against a lowercase whitelist for robust matching
        allowed_questions_lower = [q.lower() for q in allowed_questions]
        normalized_message = re.sub(r"[^\w\s]", "", message.lower().strip())
        allowed_match = any(q in normalized_message or normalized_message in q for q in allowed_questions_lower)

        # 🔹 Try fuzzy match FIRST before returning out_of_scope
        possible_match = find_closest_faq_match(message, faq_df, threshold=0.65)
        if not allowed_match and possible_match is None:
            bot_reply = (
                "Sorry, I can only answer FAQS related to the SK Council in Barangay Buhangin and events related to your interest/s.\n\n"
                "Try asking something like:\n" +
                "\n".join(f"- {q.capitalize()}" for q in random.sample(allowed_questions, 3))
            )
            log_conversation(message, "out_of_scope_faq", bot_reply, "faq_filter", user_id, username)
            return jsonify({
                "intent": "out_of_scope_faq",
                "confidence": 0.0,
                "response": bot_reply,
                "source": "faq_filter"
            })

        # 🔹 If fuzzy matched, use the closest FAQ answer
        if possible_match is not None:
            raw_answer = format_faq_answer(possible_match["bot_response"], user_text=message)
            bot_reply = finalize_reply(raw_answer, possible_match["intent"], "faq_fuzzy_match")
            return jsonify({
                "intent": possible_match["intent"],
                "confidence": 1.0,
                "response": bot_reply,
                "source": "faq_fuzzy_match"
            })

        # Continue as normal (for exact whitelist matches)
        if "intent" in faq_df.columns and predicted_intent in faq_df["intent"].values:
            # Case-insensitive intent lookup
            matched_rows = faq_df[faq_df["intent"].str.lower() == str(predicted_intent).lower()]
            responses = matched_rows["bot_response"].astype(str).tolist()
            bot_reply = generate_dynamic_reply(random.choice(responses), predicted_intent) if responses else "Sorry, I couldn’t find an answer for that."
        else:
            bot_reply = "Sorry, I couldn’t find an answer for that."
        bot_reply = personalize(bot_reply, username)
        log_conversation(message, predicted_intent, bot_reply, "faq", user_id, username)
        return jsonify({
            "intent": predicted_intent,
            "confidence": confidence,
            "response": bot_reply,
            "source": "faq",
            "categorized_interests": categorized_interests
        })

    # Event branch
    if model_type == "event":
        # If user provided interest categories, include General Administration and search per-category
        if categorized_interests:
            include_general = "General Administration"
            if include_general not in categorized_interests:
                categorized_interests.append(include_general)

            # Separate user-specific interests (exclude the general category)
            user_specific_interests = [cat for cat in categorized_interests if cat != include_general]

            # Collect recommendations for each category
            all_recs = []
            for cat in categorized_interests:
                all_recs.extend(recommend_event_all(event_df, cat))

            df_recs = pd.DataFrame(all_recs)
            if not df_recs.empty:
                df_recs = dedupe_events(df_recs)
                if requested_limit and len(df_recs) > requested_limit:
                    df_recs = df_recs.head(requested_limit)
                recommendations = df_recs.to_dict(orient="records")
                summaries = [r.get("summary", "") for r in recommendations]

                if user_specific_interests:
                    interest_text = ", ".join(user_specific_interests)
                    bot_reply = (
                        f"Based on your interests in {interest_text}, "
                        f"I’ve also included relevant general events under '{include_general}'.\n\n"
                        f"Here are {len(recommendations)} event(s):\n\n" + "\n\n".join(summaries)
                    )
                else:
                    bot_reply = (
                        f"Here are {len(recommendations)} general event(s) under '{include_general}':\n\n"
                        + "\n\n".join(summaries)
                    )
            else:
                if user_specific_interests:
                    bot_reply = f"Sorry — I couldn’t find events for your interests ({', '.join(user_specific_interests)}) or General Administration."
                else:
                    bot_reply = "Sorry — I couldn’t find any general events at the moment."

        else:
            # No specific categorized interests provided — handle general / intent-driven requests
            general_intents = {
                "general_event_recommendation", "events", "event", "general events",
                "all events", "upcoming events", "activities", "programs", "event_recommendation"
            }
            msg_lower = (message or "").lower()
            if predicted_intent in general_intents or any(kw in msg_lower for kw in ["event", "events", "upcoming", "show events", "list events"]):
                # general listing
                recommendations = recommend_event_all(event_df, limit=requested_limit)
                label = "general events"
            else:
                # treat predicted_intent as category filter
                recommendations = recommend_event_all(event_df, predicted_intent, limit=requested_limit)
                label = predicted_intent or "events"

            if recommendations:
                summaries = [r.get("summary", "") for r in recommendations]
                bot_reply = f"I found {len(recommendations)} {label}:\n\n" + "\n\n".join(summaries)
            else:
                bot_reply = f"No {label} found at the moment."

        bot_reply = personalize(bot_reply, username)

        # Optionally add SK context follow-up
        if recent_context:
            last_msg = recent_context[-1].lower()
            if "sk council" in last_msg and "event" in message.lower():
                bot_reply += "\n\nSince you asked about the SK Council earlier, would you like to see their latest PPAs or announcements?"

        # =====================================================
        # Include events from MySQL DB (filtered by interests + include "General"
        # =====================================================
        db_event_data = fetch_events_from_mysql(limit=None)  # fetch all first

        db_event_text = ""
        if db_event_data is not None and not db_event_data.empty:
            # normalize column
            db_event_data["event_type"] = db_event_data["event_type"].astype(str).str.strip()

            if categorized_interests:
                interest_keywords = [i.lower() for i in categorized_interests]

                # Filter by matching interest keywords OR event_type = "General"
                filtered_db_events = db_event_data[
                    db_event_data["event_type"].str.lower().apply(
                        lambda t: t == "general" or any(k in t for k in interest_keywords)
                    )
                ]
            else:
                # If no interests specified, show only General events
                filtered_db_events = db_event_data[
                    db_event_data["event_type"].str.lower() == "general"
                ]

            # Apply limit if requested
            if requested_limit and not filtered_db_events.empty:
                filtered_db_events = filtered_db_events.head(requested_limit)

            # Format for display
            db_event_text = format_db_events_for_reply(filtered_db_events)

            if db_event_text:
                if categorized_interests:
                    interest_text = ", ".join(categorized_interests)
                    bot_reply += (
                        "\n\n────────────────────────────\n"
                        f"📅 *Here are some upcoming events related to your interests ({interest_text}) "
                        "and general events:*\n\n"
                        f"{db_event_text}"
                    )
                else:
                    bot_reply += (
                        "\n\n────────────────────────────\n"
                        "📅 *Here are some upcoming general events from our database:*\n\n"
                        f"{db_event_text}"
                    )
        # Log final response
        log_conversation(message, predicted_intent, bot_reply, "event", user_id, username)
        return jsonify({
            "intent": predicted_intent,
            "confidence": confidence,
            "response": bot_reply,
            "source": "event",
            "categorized_interests": categorized_interests
        })

    # fallback
    bot_reply = personalize("Sorry, I couldn't process that request.", username)
    log_conversation(message, "undetermined", bot_reply, "fallback", user_id, username)
    return jsonify({
        "intent": None,
        "confidence": 0.0,
        "response": bot_reply,
        "source": "fallback"
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    message = (data.get("message") or "").lower()
    correct_intent = data.get("correct_intent")

    if message and correct_intent:
        X_new = vectorizer.transform([message])
        try:
            clf.partial_fit(X_new, [correct_intent])
        except Exception:
            pass
        log_conversation(message, correct_intent, "Corrected by user", "feedback")
        return jsonify({"status": "updated", "new_intent": correct_intent})

    return jsonify({"status": "failed", "reason": "missing message or correct_intent"}), 400


@app.route("/context/<user_id>", methods=["GET"])
def context_view(user_id):
    """View recent chat context for a specific user."""
    context = get_recent_context(user_id)
    return jsonify({"recent_messages": context})


# ========================
# Run Server
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

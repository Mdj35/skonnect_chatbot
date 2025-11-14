import os, pickle, random, re
import pandas as pd
import numpy as np
import tensorflow as tf

# transformers optional
response_generator = None
try:
    from transformers import pipeline
    response_generator = pipeline("text2text-generation", model="google/flan-t5-small")
except Exception:
    response_generator = None

from tensorflow.keras.models import load_model as tf_load_model
from db_connection import fetch_events_from_mysql

faq_model = None
event_model = None

class CompatInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        cfg = dict(config or {})
        if "batch_shape" in cfg:
            cfg.pop("batch_shape", None)
        return super().from_config(cfg, custom_objects=custom_objects)

# Try loading models (tf.keras first, fallback to keras)
try:
    faq_model = tf_load_model(
        "chatbot_model.h5",
        custom_objects={"InputLayer": CompatInputLayer},
        compile=False,
    )
    event_model = tf_load_model(
        "event_recommender_model.h5",
        custom_objects={"InputLayer": CompatInputLayer},
        compile=False,
    )
except Exception as primary_err:
    try:
        import keras as _keras  # type: ignore
        faq_model = _keras.models.load_model("chatbot_model.h5", compile=False)
        event_model = _keras.models.load_model("event_recommender_model.h5", compile=False)
    except Exception as fallback_err:
        # leave as None if loading fails
        faq_model = faq_model
        event_model = event_model

# Tokenizers / encoders
with open("tokenizer.pkl", "rb") as f:
    faq_tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    faq_label_encoder = pickle.load(f)

with open("event_tokenizer.pkl", "rb") as f:
    event_tokenizer = pickle.load(f)
with open("event_label_encoder.pkl", "rb") as f:
    event_label_encoder = pickle.load(f)

# Load FAQ dataframe and normalize
faq_df = pd.read_csv("skonnect_faq_dataset_intents.csv")
faq_df["intent"] = faq_df["intent"].astype(str).str.strip()
faq_df["patterns"] = faq_df["patterns"].astype(str).str.strip().str.lower()
faq_df["bot_response"] = faq_df["bot_response"].astype(str)
faq_df.columns = faq_df.columns.str.strip()

# Load events (CSV + DB)
csv_events = pd.read_csv("Events.csv").fillna("")
db_events = fetch_events_from_mysql()
def merge_event_sources(csv_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
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
    common_cols = ["Reference Code", "PPAs", "Description", "Recommended Category"]
    merged = pd.concat([csv_df[common_cols], db_df[common_cols]], ignore_index=True).fillna("")
    return merged

event_df = merge_event_sources(csv_events, db_events)
for col in ["Recommended Category", "PPAs", "Description", "Reference Code"]:
    if col in event_df.columns:
        event_df[col] = event_df[col].astype(str)

# shapes / encoders used by classify_message
faq_max_len, event_max_len = 20, 50

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
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
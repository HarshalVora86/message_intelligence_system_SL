"""
Streamlit application for the Message Intelligence System (spam detection).

Run:
    streamlit run app.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="✉️",
    layout="centered",
)


@st.cache_resource
def load_model():
    with open("model/spam_svc_pipeline.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("✉️ Message Intelligence System")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        message_length = st.number_input(
            "Message Length (characters)", min_value=1, max_value=500, value=75, step=1
        )
        word_count = st.number_input(
            "Word Count", min_value=1, max_value=100, value=11, step=1
        )
        num_urls = st.number_input(
            "Number of URLs", min_value=0, max_value=10, value=0, step=1
        )
        num_digits = st.number_input(
            "Number of Digits", min_value=0, max_value=50, value=0, step=1
        )
        num_special_chars = st.number_input(
            "Number of Special Characters", min_value=0, max_value=20, value=0, step=1
        )
        spam_keyword_score = st.number_input(
            "Spam Keyword Score", min_value=0, max_value=10, value=0, step=1
        )

    with col2:
        legit_keyword_score = st.number_input(
            "Legitimate Keyword Score", min_value=0, max_value=10, value=1, step=1
        )
        sender_activity_score = st.slider(
            "Sender Activity Score", min_value=0.0, max_value=100.0, value=50.0, step=0.5
        )
        sender_account_age_days = st.number_input(
            "Sender Account Age (days)", min_value=0, max_value=5000, value=275, step=1
        )
        messages_sent_last_24h = st.number_input(
            "Messages Sent (last 24h)", min_value=0, max_value=200, value=7, step=1
        )
        hour_of_day = st.number_input(
            "Hour of Day (0-23)", min_value=0, max_value=23, value=12, step=1
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda d: f"{d} - {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d]}",
            index=2,
        )

    threshold = st.slider(
        "Spam Score Threshold (lower = catch more spam)",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
    )

    submitted = st.form_submit_button(
        "🔍 Classify Message",
        width="stretch",
    )

if submitted:
    input_data = pd.DataFrame(
        {
            "message_length": [message_length],
            "word_count": [word_count],
            "num_urls": [num_urls],
            "num_digits": [num_digits],
            "num_special_chars": [num_special_chars],
            "spam_keyword_score": [spam_keyword_score],
            "legit_keyword_score": [legit_keyword_score],
            "sender_activity_score": [sender_activity_score],
            "sender_account_age_days": [sender_account_age_days],
            "messages_sent_last_24h": [messages_sent_last_24h],
            "hour_of_day": [hour_of_day],
            "day_of_week": [day_of_week],
        }
    )

    # Linear SVC: signed distance from the decision boundary (> 0 = spam side)
    score = float(model.decision_function(input_data)[0])

    st.metric("Linear SVC Spam Score", f"{score:.2f}")

    if abs(score) < 1:
        st.warning("⚠️ BORDERLINE - the message lies inside the SVM margin, send to human review")
    elif score >= threshold:
        st.error("🚫 SPAM - block or quarantine")
    else:
        st.success("✅ LEGITIMATE")

    with st.expander("View input data"):
        st.dataframe(input_data, width="stretch")

metrics_path = Path("model/metrics.json")
if metrics_path.exists():
    with st.expander("Model performance (test set)"):
        m = json.loads(metrics_path.read_text())
        st.write(
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in m.items() if k not in ("features", "target")}
        )
        st.caption(
            "Scores are perfect because the classes are almost perfectly separable "
            "in this dataset, so real-world traffic will be harder."
        )

st.divider()
st.markdown("### Project Flow")
st.code(
    "User Input → DataFrame → Saved Pipeline → Median Imputer + StandardScaler "
    "→ Linear SVC → Spam Score → Spam / Legitimate / Review"
)

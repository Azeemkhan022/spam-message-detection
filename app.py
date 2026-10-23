
from pathlib import Path

import joblib
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SpamShield | AI Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1120, #111a2e, #101827);
    color: #edf2ff;
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.92);
}

/* Sidebar readability */
[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid #293a56;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #d5e1f5 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f5f8ff !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #17243a !important;
    border-color: #425574 !important;
    color: #ffffff !important;
}

/* Hero header */
.hero {
    padding: 29px 32px;
    border-radius: 22px;
    background: linear-gradient(115deg, #182c4a, #20204a 70%, #172c42);
    border: 1px solid #344768;
    margin: 6px 0 24px;
}

.eyebrow {
    color: #67e8f9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(30px, 4vw, 43px);
    line-height: 1.15;
    margin: 12px 0;
    font-weight: 800;
}

.hero p {
    color: #d4e0f4;
    font-size: 15px;
    line-height: 1.7;
    max-width: 760px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #17243a, #131d30);
    border: 1px solid #2b3b55;
    border-radius: 16px;
    padding: 19px;
    min-height: 112px;
}

.metric-label {
    color: #b5c8e5;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
}

.muted {
    color: #b5c5df;
    font-size: 13px;
}

/* Result panels */
.result {
    border-radius: 16px;
    padding: 22px;
    margin: 15px 0 18px;
    border: 1px solid;
}

.result-spam {
    background: #351a2b;
    border-color: #a83c65;
}

.result-ham {
    background: #12352f;
    border-color: #258b70;
}

.result-title {
    font-size: 24px;
    line-height: 1.3;
    font-weight: 800;
    color: #ffffff;
}

.result-desc {
    color: #e1e9f7;
    font-size: 14px;
    margin-top: 8px;
    line-height: 1.65;
}

/* Input */
div[data-testid="stTextArea"] textarea {
    background: #101a2b !important;
    color: #f5f8ff !important;
    border: 1px solid #425574 !important;
    border-radius: 12px !important;
    line-height: 1.6 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.18) !important;
}

/* Analyze button */
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #ffffff;
    border: 0;
    border-radius: 11px;
    min-height: 50px;
    font-size: 15px;
    font-weight: 700;
    box-shadow: 0 7px 18px rgba(99, 102, 241, 0.18);
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border: 1px solid #c4b5fd;
    color: #ffffff;
    filter: brightness(1.1);
}

/* Warning message readability */
div[data-testid="stAlert"] {
    background: #172a43 !important;
    border: 1px solid #42658d !important;
    border-radius: 14px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #eaf2ff !important;
}

[data-testid="stCaptionContainer"],
.stCaption {
    color: #b5c5df !important;
}

/* Avoid unwanted empty outlines */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: transparent !important;
    box-shadow: none !important;
}

hr {
    border-color: #293952;
}

footer,
#MainMenu {
    visibility: hidden;
}

@media (max-width: 900px) {
    .hero {
        padding: 22px !important;
    }

    .hero p {
        font-size: 14px !important;
    }

    .metric-card {
        padding: 15px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ---------------- SAMPLE MESSAGES ----------------
samples = {
    "Prize scam": (
        "Congratulations! You have won a free prize. "
        "Click this link now to claim your reward!"
    ),
    "Bank details request": (
        "Your account needs verification. Please share "
        "your bank account details immediately."
    ),
    "Normal conversation": (
        "Hi, are we meeting at college tomorrow?"
    ),
    "Suspicious link": (
        "Your account has been blocked. Click this link "
        "to verify your details immediately."
    ),
}


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛡️ SpamShield")
    st.caption("MACHINE LEARNING PROJECT")
    st.divider()

    st.markdown("### 📌 Project overview")
    st.write(
        "An ML-powered tool that analyzes text messages "
        "and predicts whether they may be spam."
    )

    st.markdown("### ⚙️ Technology")
    st.markdown("""
    - 🐍 Python
    - 🧮 TF-IDF text features
    - 🧠 Scikit-learn classifier
    - 🎨 Streamlit interface
    """)

    st.divider()
    st.markdown("### 🧪 Try a sample")

    sample_choice = st.selectbox(
        "Choose an example",
        ["Write my own message", *samples.keys()],
    )

    st.caption("Choose an example to load a sample message.")


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ INTELLIGENT MESSAGE SECURITY</div>
    <h1>Spam Message Detector</h1>
    <p>
        Identify potentially unwanted messages using machine learning.
        Enter a message, analyze its language patterns, and view the
        model's prediction and probability estimates.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- TOP METRICS ----------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Analysis engine</div>
        <div class="metric-value">AI / ML</div>
        <div class="muted">Text classification</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Text processing</div>
        <div class="metric-value">TF-IDF</div>
        <div class="muted">Word pattern features</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Output</div>
        <div class="metric-value">2 Classes</div>
        <div class="muted">Spam / Not spam</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- MESSAGE INPUT ----------------
st.write("")
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown(
        '<div class="panel-title">✉️ Message analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">Paste or type the message you want to check.</div>',
        unsafe_allow_html=True,
    )

    default_message = samples.get(sample_choice, "")

    message = st.text_area(
        "Message",
        value=default_message,
        height=175,
        placeholder="Example: Congratulations! You have won a prize...",
        label_visibility="collapsed",
        key="message_input",
    )

    analyze = st.button(
        "🔍  Analyze message",
        use_container_width=True,
        type="primary",
    )

    st.caption("The message is analyzed by your locally loaded model.")


# ---------------- HOW IT WORKS ----------------
with right:
    st.markdown(
        '<div class="panel-title">📊 How it works</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("**01 · Input**")
    st.caption("Provide a text message.")

    st.markdown("**02 · Processing**")
    st.caption(
        "TF-IDF converts the text into numerical features "
        "for the trained classifier."
    )

    st.markdown("**03 · Prediction**")
    st.caption(
        "The model predicts spam or not spam and may provide "
        "probability estimates."
    )

    st.divider()
    st.markdown("**Supported input**")
    st.caption(
        "SMS, promotional text, suspicious requests, "
        "and everyday messages."
    )


# ---------------- PREDICTION ----------------
if analyze:
    if not message.strip():
        st.warning("Please enter a message before analyzing it.")

    elif not MODEL_PATH.exists():
        st.error(
            "Trained model not found. Run `python train_model.py` first."
        )

    else:
        try:
            model = load_model()
            prediction = model.predict([message])[0]
            probabilities = None

            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba([message])[0]
                probabilities = {
                    str(label): float(prob)
                    for label, prob in zip(model.classes_, raw_probs)
                }

            is_spam = str(prediction).lower() == "spam"

            st.divider()
            st.markdown("## 🧾 Analysis result")

            if is_spam:
                st.markdown("""
                <div class="result result-spam">
                    <div class="result-title">
                        🚨 Potential spam detected
                    </div>
                    <div class="result-desc">
                        The model classified this message as spam.
                        Be careful with unexpected links, prize claims,
                        payment requests, or requests for sensitive details.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="result result-ham">
                    <div class="result-title">
                        ✅ Classified as not spam
                    </div>
                    <div class="result-desc">
                        The model classified this message as a normal
                        message. This does not guarantee that it is safe.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if probabilities is not None:
                spam_prob = probabilities.get("spam", 0.0)
                ham_prob = probabilities.get("ham", 0.0)

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(
                        '<div class="metric-label">🚨 Spam estimate</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-value">{spam_prob:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(spam_prob)

                with c2:
                    st.markdown(
                        '<div class="metric-label">💬 Not-spam estimate</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="metric-value">{ham_prob:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(ham_prob)

            st.warning(
                "These are model estimates, not guarantees. A message "
                "can be fraudulent even when classified as not spam. "
                "Never share passwords, OTPs, or banking credentials "
                "in response to unexpected messages."
            )

        except Exception as error:
            st.error(f"Unable to analyze the message: {error}")


# ---------------- FOOTER ----------------
st.write("")
st.divider()

st.markdown("""
<div style="text-align:center; padding:10px 0 22px;">
    <div style="color:#dbe7f6; font-weight:700;">
        🛡️ SpamShield · Team Hustler
    </div>
    <div style="color:#aebfda; font-size:12px; margin-top:7px;">
        Built with Python · Scikit-learn · Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

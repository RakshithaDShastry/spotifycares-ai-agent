import sys
sys.path.insert(0, 'src')
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

import pandas as pd
from retrieval import ResolutionRetriever
import pipeline

load_dotenv()

def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=get_api_key())

@st.cache_resource
def get_retriever():
    resolved_pairs = pd.read_csv("data/processed/resolved_pairs.csv")
    return ResolutionRetriever(resolved_pairs)

# ---------- Page config & light styling ----------
st.set_page_config(
    page_title="SpotifyCares AI Support Agent",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deliberately minimal CSS: hide Streamlit's default menu/footer chrome and
# tint the primary button/headers. We do NOT override the page background,
# since that would break light/dark theme switching for whoever's viewing it.
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div.stButton > button[kind="primary"] {
    background-color: #1DB954;
    border-color: #1DB954;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1ed760;
    border-color: #1ed760;
}
</style>
""", unsafe_allow_html=True)

client = get_groq_client()
retriever = get_retriever()

# ---------- Header ----------
header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.markdown("<div style='font-size:3rem;'>🎧</div>", unsafe_allow_html=True)
with header_col2:
    st.title("SpotifyCares AI Support Agent")
    st.caption("An AI-powered customer support agent for intent classification, retrieval-grounded response drafting, and escalation decisions.")
st.divider()

# ---------- Sidebar ----------
if "message" not in st.session_state:
    st.session_state.message = ""

def use_example(text):
    st.session_state.message = text

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "**1. 🏷️ Classify** — identify the customer's intent\n\n"
        "**2. 🔎 Retrieve** — find similar past resolutions\n\n"
        "**3. ✍️ Draft** — write a grounded reply\n\n"
        "**4. 🚦 Escalate** — decide if a human should review"
    )
    st.divider()
    st.subheader("Try an example")

    examples = {
        "💳 Billing issue": "I was charged twice this month for my premium subscription, can you help?",
        "🐛 Technical bug": "The app keeps crashing every time I try to play a song on my phone.",
        "💡 Feature request": "Can you add a dark mode toggle for the desktop app?",
        "❓ General question": "How do I download songs for offline listening?",
    }
    for label, text in examples.items():
        st.button(label, use_container_width=True, on_click=use_example, args=(text,))

    st.divider()
    st.caption("Built with Python, Streamlit, scikit-learn (TF-IDF retrieval), and Groq (LLM inference).")
    st.caption("[View source on GitHub](https://github.com/RakshithaDShastry/spotifycares-ai-agent)")

# ---------- Main input ----------
message = st.text_area(
    "Customer message",
    key="message",
    height=120,
    placeholder="e.g. my premium subscription isn't working and I was charged twice",
)

analyze_col, clear_col = st.columns([1, 1])
with analyze_col:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
with clear_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.message = ""
        st.rerun()

# ---------- Results ----------
if analyze_clicked and message.strip():
    with st.spinner("Classifying intent..."):
        intent = pipeline.classify_intent(client, message)
    with st.container(border=True):
        st.markdown("#### 🏷️ 1. Intent Classification")
        st.info(intent)

    with st.spinner("Retrieving similar past resolutions and drafting a reply..."):
        reply, retrieved = pipeline.draft_reply(client, retriever, message, intent)

    with st.container(border=True):
        st.markdown("#### 🔎 2. Retrieved Grounding")
        if retrieved:
            for r in retrieved:
                with st.expander(f"Similarity: {r['similarity']:.2f} — {r['customer_text'][:60]}..."):
                    st.write("**Past message:**", r['customer_text'])
                    st.write("**How it was resolved:**", r['reply_text'])
        else:
            st.write("No sufficiently similar historical resolution was found.")

    with st.container(border=True):
        st.markdown("#### ✍️ 3. Drafted Reply")
        st.success(reply)

    with st.spinner("Deciding escalation..."):
        decision, reason = pipeline.decide_escalation(client, message, intent, retrieved, reply)

    with st.container(border=True):
        st.markdown("#### 🚦 4. Escalation Decision")
        if decision == "ESCALATE":
            st.warning(f"🚩 **ESCALATE** — {reason}")
        else:
            st.success(f"✅ **AUTO_HANDLE** — {reason}")

st.divider()
st.caption("An open-source portfolio project · Built with Streamlit, scikit-learn, and Groq · [GitHub](https://github.com/RakshithaDShastry/spotifycares-ai-agent)")
import sys

sys.path.insert(0, "src")

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from retrieval import ResolutionRetriever
import pipeline

load_dotenv()


def get_api_key():
    # Streamlit Cloud's secrets system first (for when it's hosted),
    # falling back to a local .env file (for local development).
    # st.secrets raises (not just returns empty) when no secrets.toml exists
    # at all, which is the normal case for local development, so we catch it.
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


st.set_page_config(
    page_title="SpotifyCares AI Support Agent",
    page_icon="🎧",
)

st.title("🎧 SpotifyCares AI Support Agent")

st.caption("An AI-powered customer support agent for intent classification, retrieval-grounded response drafting, and escalation decisions.")

client = get_groq_client()
retriever = get_retriever()

message = st.text_area(
    "Customer message",
    placeholder="e.g. my premium subscription isn't working and I was charged twice",
)

if st.button("Analyze", type="primary") and message.strip():

    with st.spinner("Classifying intent..."):
        intent = pipeline.classify_intent(client, message)

    st.subheader("1. Intent Classification")
    st.info(intent)

    with st.spinner("Retrieving similar past resolutions and drafting a reply..."):
        reply, retrieved = pipeline.draft_reply(
            client,
            retriever,
            message,
            intent,
        )

    st.subheader("2. Retrieved Grounding")

    if retrieved:
        for r in retrieved:
            with st.expander(
                f"Similarity: {r['similarity']:.2f} — {r['customer_text'][:60]}..."
            ):
                st.write("**Past message:**", r["customer_text"])
                st.write(
                    "**How it was resolved:**",
                    r["reply_text"],
                )
    else:
        st.write("No sufficiently similar historical resolution was found.")

    st.subheader("3. Drafted Reply")
    st.success(reply)

    with st.spinner("Deciding escalation..."):
        decision, reason = pipeline.decide_escalation(
            client,
            message,
            intent,
            retrieved,
            reply,
        )

    st.subheader("4. Escalation Decision")

    if decision == "ESCALATE":
        st.warning(f"🚩 **ESCALATE** — {reason}")
    else:
        st.success(f"✅ **AUTO_HANDLE** — {reason}")

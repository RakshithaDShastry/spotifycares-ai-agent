"""
Core agent pipeline: intent classification, retrieval-grounded reply
drafting, escalation decision, and LLM-as-judge reply scoring.

Uses Groq's free tier (openai/gpt-oss-120b) — see decision_log.md entry 7
for why (Gemini's free tier proved too limited for this volume).

The judge here is v2 (see decision_log.md entries 13-14): it was
deliberately kept over a v3 revision that fixed a known bias but
regressed measured agreement with human judgment on re-validation. v2's
known limitation (over-flags honestly-hedged uncertainty as a false
promise) is disclosed in reports/failure_analysis.md, not hidden.
"""

import re
import time
import random

from taxonomy_config import CATEGORIES, TAXONOMY_DEFINITIONS, build_fewshot_block
import logging

from config import (
    MODEL_NAME,
    MAX_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
    TEMPERATURE_CLASSIFY,
    TEMPERATURE_GENERATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

FEWSHOT_BLOCK = build_fewshot_block()


def strip_urls(text):
    return re.sub(r'https?://\S+', '', text).strip()


UNSAFE_PROMISE_PATTERNS = [
    r"\bwe (can|will|'ll) process (a |the |your )?refund\b",
    r"\bissue (a |the |your )?refund\b",
    r"\byou won'?t be charged\b",
    r"\bwe (can|will|'ll) merge\b",
    r"\bcombine your (account|librar)",
    r"\bwe (can|will|'ll) (waive|reverse) (the |your )?charge\b",
]


def contains_unsafe_promise(reply_text):
    """Deterministic guardrail: checks for specific unsafe promise phrasing.
    Deliberately does NOT match bare 'guarantee' — honest hedging like
    'can't guarantee a timeline' must not be flagged.
    """
    text_lower = reply_text.lower()

    for pattern in UNSAFE_PROMISE_PATTERNS:
        match = re.search(pattern, text_lower)

        if match:
            return True, match.group(0)

    return False, None


# Config values are centralized in config.py so they can be changed in one place.
def call_groq(
    groq_client,
    prompt,
    temperature=TEMPERATURE_CLASSIFY,
    max_retries=MAX_RETRIES,
):
    """Shared retry-wrapped Groq caller with exponential backoff + jitter."""
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Use the centralized retry wait limit from config.py.
            wait = min(2 ** attempt, MAX_RETRY_WAIT_SECONDS) + random.uniform(0, 1)
            logger.info(
                f"Retry {attempt + 1}/{max_retries} after {wait:.1f}s ({e})"
            )
    raise RuntimeError(f"Failed after {max_retries} attempts")


def classify_intent(groq_client, text):

    prompt = f"""You are classifying a customer support message into exactly one intent category.

Categories and definitions:

{TAXONOMY_DEFINITIONS}

Here are labeled examples:

{FEWSHOT_BLOCK}

Now classify this new message. Respond with ONLY the category name, exactly as written above — nothing else.

Message: {text}

Intent:"""

    # Classification uses the centralized deterministic temperature setting.
    result = call_groq(
        groq_client,
        prompt,
        temperature=TEMPERATURE_CLASSIFY,
    )

    cleaned_result = result.strip().rstrip(".")

    if cleaned_result in CATEGORIES:
        return cleaned_result

    logger.warning(
        f"classify_intent got unexpected output: {result!r} — "
        "falling back to 'Other / Unclear'"
    )
    return "Other / Unclear"


def draft_reply(groq_client, retriever, customer_text, intent_label, k=3):
    retrieved = retriever.retrieve(customer_text, k=k)

    if retrieved:
        grounding_block = "\n\n".join([
            f"Similar past customer message: {r['customer_text']}\nHow SpotifyCares resolved it: {strip_urls(r['reply_text'])}"
            for r in retrieved
        ])
        grounding_instruction = (
            "Below are real past customer messages and how SpotifyCares actually resolved them. "
            "Use them as a guide for tone and resolution approach ONLY where they're genuinely relevant "
            "to the current message — ignore any that don't actually match the current issue. "
            "IMPORTANT: Never include any URL/link in your reply, even if one appeared in these examples."
        )
    else:
        grounding_block = "(No sufficiently similar historical resolution was found.)"
        grounding_instruction = (
            "No relevant historical example was found. Draft a reasonable, honest reply based on general "
            "SpotifyCares support conventions — do not invent specific facts, and do not include any links."
        )

    prompt = f"""You are drafting a customer support reply for SpotifyCares.

Customer's message (classified intent: {intent_label}):
{customer_text}

{grounding_instruction}

{grounding_block}

Write a short, helpful reply in SpotifyCares' typical tone. Do not invent specific facts, ticket numbers, or links. You must NEVER promise any of the following, even implicitly — these require actions the bot cannot actually perform or guarantee:
- Merging, combining, or closing accounts
- Refunds or that a charge will/won't happen
- A specific timeline for a fix, feature, or catalog addition
- That an issue is currently being worked on right now

Instead, for these cases, say you'll pass the request to the team who can help, or ask for account details so a human can review it — without promising the outcome. Reply with ONLY the reply text, nothing else."""

    # Reply generation uses the centralized generation temperature.
    reply_text = call_groq(
        groq_client,
        prompt,
        temperature=TEMPERATURE_GENERATE,
    )
    return reply_text, retrieved


def decide_escalation(groq_client, customer_text, intent_label, retrieved, drafted_reply):
    top_similarity = retrieved[0]['similarity'] if retrieved else 0.0
    prompt = f"""You are deciding whether a customer support agent should AUTO-HANDLE or ESCALATE this case.

Customer message (intent: {intent_label}):
{customer_text}

Drafted reply:
{drafted_reply}

Best available historical grounding similarity: {top_similarity:.2f}

ESCALATE if: resolving requires an action the bot can't verify/perform (refund, payment/account changes,
identity verification — asking for DM details is fine, approving/processing something isn't); customer
explicitly asks for a human/escalation; intent is Other/Unclear or too ambiguous; similarity below 0.3 AND
issue is account-specific. Otherwise AUTO-HANDLE.

Respond in exactly this format:
DECISION: <AUTO_HANDLE or ESCALATE>
REASON: <one concise sentence>"""
    text = call_groq(groq_client, prompt, temperature=0)
    d = [l for l in text.split('\n') if l.startswith('DECISION:')]
    r = [l for l in text.split('\n') if l.startswith('REASON:')]
    decision = d[0].replace('DECISION:', '').strip() if d else 'ESCALATE'
    reason = r[0].replace('REASON:', '').strip() if r else 'Parse error, defaulting to escalate.'
    has_unsafe_promise, matched_phrase = contains_unsafe_promise(drafted_reply)

    if has_unsafe_promise:
        logger.warning(
            f"Guardrail override: unsafe promise phrase "
            f"'{matched_phrase}' forced ESCALATE"
        )
        decision = "ESCALATE"
        reason = (
            f"Deterministic guardrail override: reply contains unsafe "
            f"promise phrasing ('{matched_phrase}'), regardless of the "
            f"policy model's decision."
        )

    return decision, reason


def judge_reply(groq_client, customer_text, intent_label, drafted_reply):
    """v2 judge — see module docstring for why this version was kept."""
    prompt = f"""You are evaluating a customer support reply. Follow these steps in order.

Customer message (intent: {intent_label}):
{customer_text}

Drafted reply:
{drafted_reply}

STEP 1: List every specific claim, promise, or explanation the bot cannot actually verify or guarantee.
Asking for DM account details is NOT an unverifiable claim.

STEP 2: Score 1-5. If STEP 1 found ANY unverifiable claim, score is AT MOST 2 regardless of tone.
Otherwise score normally (5=excellent, 3=reasonable/generic, 1=irrelevant/wrong tone).

Respond in exactly this format:
UNVERIFIABLE_CLAIMS: <list or "none">
SCORE: <1-5>
JUSTIFICATION: <one sentence>"""
    text = call_groq(groq_client, prompt, temperature=0)
    lines = text.split('\n')
    s = [l for l in lines if l.startswith('SCORE:')]
    j = [l for l in lines if l.startswith('JUSTIFICATION:')]
    try:
        score = int(s[0].replace('SCORE:', '').strip())
    except (IndexError, ValueError):
        score = None
    justification = j[0].replace('JUSTIFICATION:', '').strip() if j else 'Parse error.'
    return score, justification
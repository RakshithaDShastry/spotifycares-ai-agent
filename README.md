# SpotifyCares AI Support Agent

🔗 **[Try the live demo](https://spotifycares-ai-agent-zdceekkuybonc89jwzuzrq.streamlit.app/)**

An AI support agent for **SpotifyCares** (Spotify's Twitter support
account), built on Kaggle's "Customer Support on Twitter" dataset. It
classifies incoming customer messages, drafts a reply grounded in how
SpotifyCares has historically resolved similar issues, and decides whether
to auto-handle or escalate to a human — with a stated reason.

**The full report — problem framing, results vs. baselines, failure
analysis, and what's misleading about the headline numbers — is in
`reports/SpotifyCares_Agent_Report.docx`.** This README covers setup and
reproduction only.

## Quickstart — reproduce headline results in under 15 minutes

This is the **fast path**: it recomputes all headline metrics from saved
model outputs. No API key needed, runs in seconds.

```bash
git clone <repo-url>
cd spotifycares-ai-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pandas scikit-learn   # only these two are needed for this path

python scripts/compute_metrics.py
```

This prints: the intent classifier's accuracy vs. both baselines (majority
class, TF-IDF + Logistic Regression) with a full per-category breakdown,
and the reply-drafting/escalation pipeline's judge scores and escalation
split — reproducing the exact numbers cited in `reports/SpotifyCares_Agent_Report.docx`.

## Optional: regenerate everything from scratch (calls real LLM APIs)

This path re-runs the actual pipeline — intent classification, retrieval,
reply drafting, escalation, and judging — by calling Groq's API. It's
**slower** (the full 200-example classification + 50-example pipeline eval
takes on the order of 20-30 minutes due to API pacing) and needs a free
Groq API key (get one at console.groq.com, no card required). This is not
part of the 15-minute reproduction promise above — it's provided for
anyone who wants to verify the pipeline itself, not just its saved outputs.

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

The full pipeline logic lives in `src/pipeline.py` (`classify_intent`,
`draft_reply`, `decide_escalation`, `judge_reply`) and `src/retrieval.py`
(`ResolutionRetriever`). These were developed and iterated on in a Colab
notebook (dataset download, retrieval index, and every debugging/fix step
is preserved there); the exported `.py` files here are the exact,
verified-matching final versions — see `reports/decision_log.md` for the
full development history, including two real bugs found and fixed during
evaluation (a URL-fabrication issue in reply generation, and an LLM-judge
miscalibration found via blind human validation).

## Data

- **Primary:** Kaggle `thoughtvector/customer-support-on-twitter` (~3M
  tweets). We use a subsample: SpotifyCares' ~26k conversation-opening
  customer messages and ~41k historical resolution pairs, not the full
  dataset (this is by design — see `reports/decision_log.md`, entry 2).
- **Golden evaluation set:** `data/processed/golden_set.csv` — 200
  hand-labelled examples. Sampled via `random_state=99` from the 26,068
  SpotifyCares conversation openers, then hand-labelled by the author
  against the taxonomy in `reports/taxonomy.md`. The taxonomy itself was
  refined *during* labelling (e.g. "Content Availability" split out from
  general technical bugs) once real examples showed the distinction
  mattered — see `reports/taxonomy.md`'s revision note for detail.

## Repo structure

```
data/
  raw/            # not committed — fetched via kagglehub (see full pipeline path)
  processed/      # golden_set.csv, golden_set_final_results.csv,
                   # baseline_results.csv, final_eval_results.csv
src/
  data_prep.py    # builds the SpotifyCares subsets from the raw dataset
  retrieval.py    # TF-IDF grounding retrieval (ResolutionRetriever)
  taxonomy_config.py   # categories, definitions, few-shot examples
  pipeline.py     # classify_intent, draft_reply, decide_escalation, judge_reply
  baselines.py    # majority-class and TF-IDF+LogisticRegression baselines
scripts/
  compute_metrics.py    # fast reproduction path (see Quickstart)
reports/
  SpotifyCares_Agent_Report.docx # the actual report — read this first
  taxonomy.md           # intent taxonomy with grounding evidence
  decision_log.md        # 14 non-obvious decisions and why
  failure_analysis.md    # top 5 failure patterns with real examples
```

## Rules compliance

AI coding assistance was used throughout (disclosed openly, not hidden) —
every design decision, its reasoning, and every bug found along the way is
recorded in `reports/decision_log.md` and `reports/failure_analysis.md`,
specifically so the author can explain and modify any part of this on
request.

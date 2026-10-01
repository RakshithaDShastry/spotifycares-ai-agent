"""
Central place for every tunable value in the pipeline.

Change a setting here once, instead of hunting through
multiple files.
"""

# LLM settings
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE_CLASSIFY = 0.0
TEMPERATURE_GENERATE = 0.3
MAX_RETRIES = 6
MAX_RETRY_WAIT_SECONDS = 30

# Retrieval settings
RETRIEVAL_K = 3
RETRIEVAL_MIN_SIMILARITY = 0.25
RETRIEVAL_MIN_WORD_COUNT = 5

TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 2
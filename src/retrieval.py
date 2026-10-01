"""
Retrieval: finds real historical (customer message, SpotifyCares reply)
pairs similar to a new customer message, to ground reply generation.

Uses TF-IDF + cosine similarity rather than neural embeddings — a
deliberate simplicity/speed tradeoff given the project timeline; see
decision_log.md entry 9 for the reasoning and known limitation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    RETRIEVAL_K,
    RETRIEVAL_MIN_SIMILARITY,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
)


class ResolutionRetriever:
    def __init__(self, resolved_pairs, max_features=10000, ngram_range=(1, 2), min_df=2):
        self.resolved_pairs = resolved_pairs.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=TFIDF_MIN_DF,
        )
        self.matrix = self.vectorizer.fit_transform(self.resolved_pairs['text_customer'])

    def retrieve(self,query,k=RETRIEVAL_K,min_similarity=RETRIEVAL_MIN_SIMILARITY,):
        
        """
        Returns up to k historical (customer_text, reply_text, similarity)
        matches above min_similarity, excluding near-exact matches of the
        query itself (similarity >= 0.999) to prevent leakage when the
        query is a message that's already in the historical corpus.
        """
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_idx:
            if similarities[idx] >= 0.999:
                continue
            if similarities[idx] < min_similarity:
                break
            results.append({
                'customer_text': self.resolved_pairs.iloc[idx]['text_customer'],
                'reply_text': self.resolved_pairs.iloc[idx]['text_reply'],
                'similarity': float(similarities[idx]),
            })
            if len(results) == k:
                break
        return results

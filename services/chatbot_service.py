"""
Hybrid Retail Chatbot Service.

Combines:
1. Rule-based intent matching for exact FAQ keyword/pattern triggers.
2. TF-IDF Cosine Similarity / Classifier Fallback for dynamic customer queries.
"""

import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.nlp_service import TextPreprocessor

INTENTS_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"


class HybridChatbotService:
    """
    Hybrid Retail FAQ & Customer Support Chatbot.
    """

    def __init__(self, intents_path: Optional[Union[str, Path]] = None):
        self.intents_path = Path(intents_path) if intents_path else INTENTS_FILE_PATH
        self.preprocessor = TextPreprocessor()
        self.intents: List[Dict[str, Any]] = []
        self.patterns: List[str] = []
        self.pattern_tags: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.load_intents()
        self._build_tfidf_fallback_index()

    def load_intents(self):
        """Load 20 retail FAQs from JSON schema file."""
        if not self.intents_path.exists():
            raise FileNotFoundError(f"Intents schema file not found at: {self.intents_path}")

        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.intents = data.get("intents", [])

        print(f"[HybridChatbot] Loaded {len(self.intents)} FAQ intent categories.")

    def _build_tfidf_fallback_index(self):
        """Build TF-IDF matrix across all intent patterns for dynamic fallback matching."""
        self.patterns = []
        self.pattern_tags = []

        for intent in self.intents:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                clean_p = self.preprocessor.preprocess(pattern)
                if clean_p:
                    self.patterns.append(clean_p)
                    self.pattern_tags.append(tag)

        if self.patterns:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def rule_based_match(self, query: str) -> Optional[Tuple[str, float, str]]:
        """
        Rule-based keyword & pattern matcher.

        :param query: Customer raw text query string.
        :return: Tuple of (tag, confidence, response) or None if no keyword matches.
        """
        clean_query = self.preprocessor.preprocess(query)
        query_words = set(clean_query.split())
        raw_query_lower = query.lower()

        best_tag = None
        highest_score = 0.0
        best_intent_obj = None

        for intent in self.intents:
            score = 0.0
            # 1. Check exact keyphrases
            for kw in intent.get("keywords", []):
                kw_lower = kw.lower()
                if kw_lower in raw_query_lower:
                    score += 2.0
                elif any(word == kw_lower for word in query_words):
                    score += 1.0

            # 2. Check pattern similarity
            for p in intent.get("patterns", []):
                p_clean = self.preprocessor.preprocess(p)
                p_words = set(p_clean.split())
                overlap = len(query_words.intersection(p_words))
                if overlap > 0:
                    score += overlap * 0.5

            if score > highest_score:
                highest_score = score
                best_tag = intent["tag"]
                best_intent_obj = intent

        if highest_score >= 1.5 and best_intent_obj:
            confidence = min(0.99, 0.70 + (highest_score * 0.05))
            response = random.choice(best_intent_obj["responses"])
            return best_tag, confidence, response

        return None

    def tfidf_fallback_match(self, query: str) -> Tuple[str, float, str]:
        """
        TF-IDF Cosine Similarity Fallback for dynamic/unstructured customer queries.

        :param query: Preprocessed customer query string.
        :return: Tuple of (tag, confidence, response).
        """
        clean_query = self.preprocessor.preprocess(query)
        if not clean_query or self.vectorizer is None or self.tfidf_matrix is None:
            return self._default_fallback_response()

        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim >= 0.25:
            matched_tag = self.pattern_tags[max_idx]
            # Find matching intent object
            intent_obj = next((i for i in self.intents if i["tag"] == matched_tag), None)
            if intent_obj:
                response = random.choice(intent_obj["responses"])
                return matched_tag, round(max_sim, 4), response

        return self._default_fallback_response()

    def _default_fallback_response(self) -> Tuple[str, float, str]:
        """Default response when query confidence is low."""
        fallback_msg = (
            "I'm sorry, I didn't quite catch that. You can ask me about our return policy, "
            "shipping options, store hours, order tracking, or contact our support team at 1-800-SMART-RETAIL."
        )
        return "general_fallback", 0.0, fallback_msg

    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Main query handler. Executes Rule-Based matching first, then falls back to TF-IDF classifier.

        :param query: Customer question/message.
        :return: Result dictionary with response, matched intent, method, and confidence.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "matched_intent": "empty_query",
                "matching_method": "none",
                "confidence": 0.0,
                "response": "Please enter a valid message or question.",
            }

        # Step 1: Rule-Based exact keyword match
        rule_result = self.rule_based_match(query)
        if rule_result is not None:
            tag, confidence, response = rule_result
            return {
                "query": query,
                "matched_intent": tag,
                "matching_method": "rule_based",
                "confidence": round(confidence, 4),
                "response": response,
            }

        # Step 2: TF-IDF Cosine Similarity Fallback
        tag, confidence, response = self.tfidf_fallback_match(query)
        method = "tfidf_fallback" if tag != "general_fallback" else "fallback_default"

        return {
            "query": query,
            "matched_intent": tag,
            "matching_method": method,
            "confidence": round(confidence, 4),
            "response": response,
        }

    def get_all_intents(self) -> List[Dict[str, Any]]:
        """Return list of all 20 FAQ intents."""
        return [
            {
                "tag": i["tag"],
                "keywords": i.get("keywords", []),
                "sample_patterns_count": len(i.get("patterns", [])),
            }
            for i in self.intents
        ]


# Singleton instance
_chatbot_service: Optional[HybridChatbotService] = None


def get_chatbot_service() -> HybridChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = HybridChatbotService()
    return _chatbot_service

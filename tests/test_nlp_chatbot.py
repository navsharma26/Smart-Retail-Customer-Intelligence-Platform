"""
Unit test suite for Phase 3 NLP & Hybrid Chatbot Services.
"""

import json
from pathlib import Path
import unittest

from services.chatbot_service import HybridChatbotService
from services.nlp_service import (
    SentimentAnalyzerService,
    TextPreprocessor,
    preprocess_text,
)


class TestNLPAndChatbotServices(unittest.TestCase):
    """Test suite verifying TextPreprocessor, SentimentAnalyzerService, and HybridChatbotService."""

    def setUp(self):
        """Initialize services for testing."""
        self.preprocessor = TextPreprocessor()
        self.sentiment_service = SentimentAnalyzerService()
        self.chatbot_service = HybridChatbotService()

    def test_text_preprocessor(self):
        """Test lowercasing, punctuation removal, stopword filtering, and lemmatization."""
        raw_text = "I LOVED these shoes! Highly recommended for buying."
        clean = self.preprocessor.preprocess(raw_text)

        self.assertNotIn("!", clean)
        self.assertNotIn("these", clean)  # Stopword removed
        self.assertIn("love", clean)  # Lemmatized
        self.assertIn("shoe", clean)  # Lemmatized

        # Convenience function
        self.assertEqual(preprocess_text("Running fast!"), "run fast")

    def test_sentiment_analyzer_positive(self):
        """Test sentiment classification on positive review."""
        review = "Exceptional product quality! Super fast delivery and wonderful packaging."
        res = self.sentiment_service.analyze_sentiment(review)

        self.assertEqual(res["sentiment"], "positive")
        self.assertGreaterEqual(res["confidence_score"], 0.5)
        self.assertIn("class_probabilities", res)

    def test_sentiment_analyzer_negative(self):
        """Test sentiment classification on negative review."""
        review = "Terrible quality. The zipper broke on the first day of use. Horrible experience."
        res = self.sentiment_service.analyze_sentiment(review)

        self.assertEqual(res["sentiment"], "negative")
        self.assertGreaterEqual(res["confidence_score"], 0.5)

    def test_intents_json_count(self):
        """Verify data/intents.json contains exactly 20 retail FAQ categories."""
        intents = self.chatbot_service.get_all_intents()
        self.assertEqual(len(intents), 20)

    def test_chatbot_rule_based_matching(self):
        """Test rule-based exact keyword matching in HybridChatbotService."""
        query = "What is your return policy for money back?"
        res = self.chatbot_service.get_response(query)

        self.assertEqual(res["matched_intent"], "return_policy")
        self.assertEqual(res["matching_method"], "rule_based")
        self.assertGreaterEqual(res["confidence"], 0.7)
        self.assertTrue(len(res["response"]) > 0)

    def test_chatbot_tfidf_fallback_matching(self):
        """Test TF-IDF similarity fallback for dynamic query phrasing."""
        query = "How many days will it take for standard freight delivery to arrive?"
        res = self.chatbot_service.get_response(query)

        self.assertIn(res["matched_intent"], ["shipping_info", "order_tracking"])
        self.assertIn(res["matching_method"], ["rule_based", "tfidf_fallback"])
        self.assertTrue(len(res["response"]) > 0)


if __name__ == "__main__":
    unittest.main()

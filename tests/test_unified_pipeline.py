"""
Unit test suite for Phase 4 Unified Platform Pipeline & FastAPI Endpoints.
"""

import unittest
import cv2
import numpy as np

from services.pipeline import UnifiedPlatformPipeline, get_unified_pipeline


class TestUnifiedPipeline(unittest.TestCase):
    """Test suite verifying UnifiedPlatformPipeline operations."""

    def setUp(self):
        """Initialize pipeline and synthetic test data."""
        self.pipeline = get_unified_pipeline()

        # Synthetic product image (224x224 RGB)
        self.product_image = np.full((224, 224, 3), 150, dtype=np.uint8)
        cv2.rectangle(self.product_image, (30, 30), (190, 190), (100, 200, 50), -1)

        # Synthetic facial image (300x300 RGB)
        self.face_image = np.full((300, 300, 3), 210, dtype=np.uint8)
        cv2.circle(self.face_image, (150, 150), 60, (120, 120, 120), -1)

    def test_pipeline_classify_product(self):
        """Test unified product classification."""
        res = self.pipeline.classify_product(self.product_image)
        self.assertIn("predicted_category", res)
        self.assertIn("confidence_score", res)
        self.assertEqual(len(res["class_probabilities"]), 5)

    def test_pipeline_recognize_face(self):
        """Test unified face recognition execution."""
        res = self.pipeline.recognize_face(self.face_image)
        self.assertIn("matched", res)
        self.assertIn("confidence_score", res)

    def test_pipeline_analyze_sentiment(self):
        """Test unified sentiment analysis execution."""
        text = "This retail store experience was fantastic and very fast!"
        res = self.pipeline.analyze_sentiment(text)
        self.assertIn("sentiment", res)
        self.assertIn("confidence_score", res)
        self.assertEqual(res["sentiment"], "positive")

    def test_pipeline_chatbot_query(self):
        """Test unified chatbot query execution."""
        query = "What is your return policy?"
        res = self.pipeline.chatbot_query(query)
        self.assertEqual(res["matched_intent"], "return_policy")
        self.assertIn("response", res)

    def test_pipeline_dashboard_stats(self):
        """Test unified dashboard statistics compilation."""
        stats = self.pipeline.get_dashboard_stats()
        self.assertEqual(stats["status"], "success")
        self.assertIn("customer_analytics", stats)
        self.assertIn("sentiment_telemetry", stats)
        self.assertIn("models_loaded", stats)


if __name__ == "__main__":
    unittest.main()

"""
Unit test suite for Phase 2 CV Services (ProductClassifierService & FaceRecognitionDBService).
"""

import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from services.cv_service import (
    FaceRecognitionDBService,
    ProductClassifierService,
)


class TestCVService(unittest.TestCase):
    """Test suite verifying Product Classifier and Face Recognition DB modules."""

    def setUp(self):
        """Set up synthetic test images and temp file paths."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_face_db.pkl"

        # Synthetic product test image (224x224 RGB)
        self.product_image = np.full((224, 224, 3), 180, dtype=np.uint8)
        cv2.rectangle(self.product_image, (40, 40), (180, 180), (50, 100, 200), -1)

        # Synthetic face test image (300x300 RGB with face representation)
        self.face_image1 = np.full((300, 300, 3), 220, dtype=np.uint8)
        cv2.circle(self.face_image1, (150, 150), 70, (140, 140, 140), -1)
        cv2.circle(self.face_image1, (125, 130), 8, (30, 30, 30), -1)
        cv2.circle(self.face_image1, (175, 130), 8, (30, 30, 30), -1)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_product_classifier_output(self):
        """Test ProductClassifierService prediction structure."""
        service = ProductClassifierService()
        result = service.classify(self.product_image)

        self.assertIn("predicted_category", result)
        self.assertIn(result["predicted_category"], service.categories)
        self.assertIn("confidence_score", result)
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)
        self.assertEqual(len(result["class_probabilities"]), 5)

    def test_face_db_registration_and_identification(self):
        """Test FaceRecognitionDBService customer registration, identification, and visit logging."""
        face_db = FaceRecognitionDBService(db_path=self.temp_db_path)

        # 1. Register customer
        reg_res = face_db.register_customer(
            customer_id="CUST-101",
            name="Alice Smith",
            image_bgr=self.face_image1,
        )
        self.assertEqual(reg_res["status"], "success")
        self.assertEqual(reg_res["customer_id"], "CUST-101")
        self.assertTrue(self.temp_db_path.exists())

        # 2. Identify customer with same image
        id_res = face_db.identify_customer(self.face_image1, tolerance=0.6)
        self.assertTrue(id_res["matched"])
        self.assertEqual(id_res["customer_id"], "CUST-101")
        self.assertEqual(id_res["customer_name"], "Alice Smith")
        self.assertGreaterEqual(id_res["total_visits"], 2)  # 1 initial + 1 identified visit

        # 3. Reload database from PKL file to verify persistence
        reloaded_db = FaceRecognitionDBService(db_path=self.temp_db_path)
        all_customers = reloaded_db.get_all_customers()
        self.assertEqual(len(all_customers), 1)
        self.assertEqual(all_customers[0]["customer_id"], "CUST-101")
        self.assertEqual(all_customers[0]["total_visits"], 2)


if __name__ == "__main__":
    unittest.main()

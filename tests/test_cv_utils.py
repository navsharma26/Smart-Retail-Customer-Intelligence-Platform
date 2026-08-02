"""
Unit test suite for Computer Vision utilities.
"""

import unittest
import cv2
import numpy as np

from services.cv_utils import (
    CVProcessor,
    apply_canny_edge_detection,
    convert_to_grayscale,
    detect_faces_haar,
    process_image_pipeline,
    resize_image,
)


class TestCVUtils(unittest.TestCase):
    """Test suite verifying OpenCV processing functions."""

    def setUp(self):
        """Create a synthetic RGB image for testing."""
        # Create a 400x600 synthetic color image with a drawn circle representing a face
        self.test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        # Background color
        self.test_image[:] = (200, 200, 200)
        # Draw face circle
        cv2.circle(self.test_image, (300, 200), 80, (150, 150, 150), -1)
        # Draw eyes
        cv2.circle(self.test_image, (270, 180), 10, (50, 50, 50), -1)
        cv2.circle(self.test_image, (330, 180), 10, (50, 50, 50), -1)

    def test_load_image_numpy(self):
        """Test loading from numpy array."""
        processor = CVProcessor()
        loaded = processor.load_image(self.test_image)
        self.assertEqual(loaded.shape, self.test_image.shape)

    def test_load_image_bytes(self):
        """Test loading from raw bytes."""
        success, encoded = cv2.imencode(".jpg", self.test_image)
        self.assertTrue(success)
        image_bytes = encoded.tobytes()

        processor = CVProcessor()
        loaded = processor.load_image(image_bytes)
        self.assertEqual(loaded.shape[2], 3)

    def test_resize_image_aspect_ratio(self):
        """Test resizing image while preserving aspect ratio."""
        resized = resize_image(self.test_image, width=300)
        self.assertEqual(resized.shape[1], 300)
        self.assertEqual(resized.shape[0], 200)  # 400 * (300/600) = 200

    def test_convert_to_grayscale(self):
        """Test converting image to 2D grayscale."""
        gray = convert_to_grayscale(self.test_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape[0], self.test_image.shape[0])
        self.assertEqual(gray.shape[1], self.test_image.shape[1])

    def test_apply_canny_edge_detection(self):
        """Test Canny edge detection filter."""
        edges = apply_canny_edge_detection(self.test_image, 50, 150)
        self.assertEqual(len(edges.shape), 2)
        self.assertEqual(edges.shape[0], self.test_image.shape[0])
        self.assertEqual(edges.shape[1], self.test_image.shape[1])
        # Verify edge pixel values are binary (0 or 255)
        unique_vals = set(np.unique(edges))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_detect_faces_haar(self):
        """Test Haar cascade face detection call structure."""
        faces = detect_faces_haar(self.test_image)
        self.assertIsInstance(faces, list)

    def test_process_image_pipeline(self):
        """Test complete vision processing pipeline."""
        result = process_image_pipeline(self.test_image, target_width=300)
        self.assertIn("original_dimensions", result)
        self.assertIn("processed_dimensions", result)
        self.assertIn("faces_detected_count", result)
        self.assertIn("bounding_boxes", result)
        self.assertIn("image_bgr", result)
        self.assertIn("image_gray", result)
        self.assertIn("image_edges", result)
        self.assertEqual(result["processed_dimensions"]["width"], 300)


if __name__ == "__main__":
    unittest.main()

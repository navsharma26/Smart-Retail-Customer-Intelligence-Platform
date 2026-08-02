"""
Computer Vision Utilities for Smart Retail & Customer Intelligence Platform.

Provides image processing utilities using OpenCV:
- Image loading (from path, bytes, or numpy array)
- Image resizing with aspect ratio preservation
- Grayscale conversion
- Canny edge detection
- Face detection using Haar Cascades
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class CVProcessor:
    """
    OpenCV-based Computer Vision Processor class.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the CVProcessor with a Haar Cascade classifier.

        :param cascade_path: Path to custom Haar Cascade XML file.
                             Defaults to OpenCV's built-in frontal face cascade.
        """
        if cascade_path is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar cascade XML file not found at path: {cascade_path}"
            )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade classifier from {cascade_path}"
            )

    @staticmethod
    def load_image(input_data: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """
        Convert input data (file path, raw bytes, or numpy array) into an OpenCV BGR image matrix.

        :param input_data: File path, byte array, or existing NumPy image array.
        :return: OpenCV BGR NumPy image array.
        :raises ValueError: If image cannot be read or decoded.
        """
        if isinstance(input_data, np.ndarray):
            if input_data.size == 0:
                raise ValueError("Input NumPy array is empty.")
            return input_data.copy()

        if isinstance(input_data, (str, Path)):
            path_str = str(input_data)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image file does not exist: {path_str}")
            image = cv2.imread(path_str)
            if image is None:
                raise ValueError(f"OpenCV failed to read image from path: {path_str}")
            return image

        if isinstance(input_data, bytes):
            nparr = np.frombuffer(input_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV failed to decode image from raw bytes.")
            return image

        raise TypeError(
            f"Unsupported input type for image loading: {type(input_data)}"
        )

    @staticmethod
    def resize_image(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        inter: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio if only one dimension is specified.

        :param image: Input NumPy image array.
        :param width: Target width (optional).
        :param height: Target height (optional).
        :param inter: Interpolation flag for cv2.resize.
        :return: Resized NumPy image array.
        """
        if width is None and height is None:
            return image.copy()

        (h, w) = image.shape[:2]

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

        return cv2.resize(image, dim, interpolation=inter)

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert BGR/RGB image to single-channel Grayscale image.

        :param image: Input NumPy image array.
        :return: Grayscale image array.
        """
        if len(image.shape) == 2:
            return image.copy()

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_canny_edge_detection(
        self,
        image: np.ndarray,
        threshold1: float = 100.0,
        threshold2: float = 200.0,
        l2gradient: bool = False,
    ) -> np.ndarray:
        """
        Apply Canny Edge Detection algorithm to an image.

        :param image: Input image array (Grayscale or BGR).
        :param threshold1: First threshold for hysteresis procedure.
        :param threshold2: Second threshold for hysteresis procedure.
        :param l2gradient: Flag indicating whether L2 norm should be used for gradient magnitude.
        :return: Binary edge map image.
        """
        gray = self.convert_to_grayscale(image)
        # Apply Gaussian Blur to reduce noise before Canny detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred, threshold1=threshold1, threshold2=threshold2, L2gradient=l2gradient
        )
        return edges

    def detect_faces_haar(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[Dict[str, int]]:
        """
        Detect facial bounding boxes using OpenCV Haar Cascades.

        :param image: Input image array.
        :param scale_factor: Parameter specifying how much the image size is reduced at each image scale.
        :param min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have.
        :param min_size: Minimum possible object size.
        :return: List of dictionaries containing bounding box coordinates: [{'x': x, 'y': y, 'w': w, 'h': h}]
        """
        gray = self.convert_to_grayscale(image)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )

        bounding_boxes = []
        for x, y, w, h in faces:
            bounding_boxes.append(
                {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            )

        return bounding_boxes

    def process_pipeline(
        self,
        input_data: Union[str, Path, bytes, np.ndarray],
        target_width: Optional[int] = 800,
        canny_threshold1: float = 100.0,
        canny_threshold2: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Execute full CV Pipeline: Load -> Resize -> Grayscale -> Canny Edges -> Face Bounding Boxes.

        :param input_data: Raw input image (path, bytes, or numpy array).
        :param target_width: Width to resize image to (preserves aspect ratio).
        :param canny_threshold1: Canny edge low threshold.
        :param canny_threshold2: Canny edge high threshold.
        :return: Pipeline execution dictionary with visual maps and face metadata.
        """
        original_image = self.load_image(input_data)
        orig_h, orig_w = original_image.shape[:2]

        resized_image = self.resize_image(original_image, width=target_width)
        res_h, res_w = resized_image.shape[:2]

        gray_image = self.convert_to_grayscale(resized_image)
        canny_edges = self.apply_canny_edge_detection(
            resized_image, threshold1=canny_threshold1, threshold2=canny_threshold2
        )

        faces = self.detect_faces_haar(resized_image)

        return {
            "original_dimensions": {"width": orig_w, "height": orig_h},
            "processed_dimensions": {"width": res_w, "height": res_h},
            "faces_detected_count": len(faces),
            "bounding_boxes": faces,
            "image_bgr": resized_image,
            "image_gray": gray_image,
            "image_edges": canny_edges,
        }


# Convenience top-level functions for direct invocation
_default_processor: Optional[CVProcessor] = None


def get_cv_processor() -> CVProcessor:
    """Singleton getter for default CVProcessor."""
    global _default_processor
    if _default_processor is None:
        _default_processor = CVProcessor()
    return _default_processor


def resize_image(
    image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    """Top-level utility for resizing an image."""
    return CVProcessor.resize_image(image, width=width, height=height)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Top-level utility for converting an image to grayscale."""
    return CVProcessor.convert_to_grayscale(image)


def apply_canny_edge_detection(
    image: np.ndarray, threshold1: float = 100.0, threshold2: float = 200.0
) -> np.ndarray:
    """Top-level utility for Canny edge detection."""
    processor = get_cv_processor()
    return processor.apply_canny_edge_detection(
        image, threshold1=threshold1, threshold2=threshold2
    )


def detect_faces_haar(
    image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5
) -> List[Dict[str, int]]:
    """Top-level utility for Haar face detection."""
    processor = get_cv_processor()
    return processor.detect_faces_haar(
        image, scale_factor=scale_factor, min_neighbors=min_neighbors
    )


def process_image_pipeline(
    input_data: Union[str, Path, bytes, np.ndarray], target_width: Optional[int] = 800
) -> Dict[str, Any]:
    """Top-level utility for executing the complete CV pipeline."""
    processor = get_cv_processor()
    return processor.process_pipeline(input_data, target_width=target_width)

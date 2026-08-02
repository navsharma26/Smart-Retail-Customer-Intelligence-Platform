"""
Computer Vision Router endpoints.
"""

import base64
from typing import List, Optional
import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.schemas import (
    CVPipelineResponse,
    CustomerFaceRecognitionResponse,
    CustomerProfile,
    CustomerRegistrationResponse,
    FaceDetectionResponse,
    ProductClassificationResponse,
)
from services.cv_service import (
    get_face_recognition_db_service,
    get_product_classifier_service,
)
from services.cv_utils import get_cv_processor

router = APIRouter(prefix="/cv", tags=["Computer Vision Services"])


def _to_base64_jpeg(image_np) -> str:
    """Encode OpenCV NumPy image matrix to base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image_np)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


@router.post(
    "/detect-faces",
    response_model=FaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect faces in an uploaded image using OpenCV Haar Cascades",
)
async def detect_faces(file: UploadFile = File(...)):
    """Accepts an uploaded image file and returns face bounding boxes."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image.",
        )

    try:
        contents = await file.read()
        processor = get_cv_processor()
        image = processor.load_image(contents)

        h, w = image.shape[:2]
        bounding_boxes = processor.detect_faces_haar(image)

        return FaceDetectionResponse(
            status="success",
            faces_count=len(bounding_boxes),
            bounding_boxes=bounding_boxes,
            image_dimensions={"width": w, "height": h},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing error: {str(e)}",
        )


@router.post(
    "/process-pipeline",
    response_model=CVPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full Computer Vision pipeline (Resize, Grayscale, Canny Edges, Face Detection)",
)
async def process_pipeline(
    file: UploadFile = File(...),
    target_width: Optional[int] = Query(
        800, ge=100, le=3840, description="Target width for aspect-ratio resize"
    ),
):
    """Executes image preprocessing & feature extraction pipeline."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image.",
        )

    try:
        contents = await file.read()
        processor = get_cv_processor()
        pipeline_result = processor.process_pipeline(
            contents, target_width=target_width
        )

        gray_b64 = _to_base64_jpeg(pipeline_result["image_gray"])
        edges_b64 = _to_base64_jpeg(pipeline_result["image_edges"])

        return CVPipelineResponse(
            status="success",
            original_dimensions=pipeline_result["original_dimensions"],
            processed_dimensions=pipeline_result["processed_dimensions"],
            faces_detected_count=pipeline_result["faces_detected_count"],
            bounding_boxes=pipeline_result["bounding_boxes"],
            gray_image_base64=gray_b64,
            edges_image_base64=edges_b64,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV Pipeline error: {str(e)}",
        )


@router.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify retail product image into 5 categories using MobileNetV2",
)
async def classify_product(file: UploadFile = File(...)):
    """Classifies uploaded image into categories: shoes, bags, electronics, clothing, groceries."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image.",
        )

    try:
        contents = await file.read()
        cv_proc = get_cv_processor()
        image_bgr = cv_proc.load_image(contents)

        classifier = get_product_classifier_service()
        result = classifier.classify(image_bgr)

        return ProductClassificationResponse(
            status="success",
            predicted_category=result["predicted_category"],
            confidence_score=result["confidence_score"],
            confidence_percentage=result["confidence_percentage"],
            class_probabilities=result["class_probabilities"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product classification error: {str(e)}",
        )


@router.post(
    "/register-customer",
    response_model=CustomerRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new customer face profile into face_db.pkl",
)
async def register_customer(
    customer_id: str = Form(..., description="Unique customer ID (e.g. CUST-1001)"),
    name: str = Form(..., description="Customer full name"),
    file: UploadFile = File(..., description="Customer facial image"),
):
    """Extracts facial encoding and registers customer profile into persistent PKL database."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image.",
        )

    try:
        contents = await file.read()
        cv_proc = get_cv_processor()
        image_bgr = cv_proc.load_image(contents)

        face_service = get_face_recognition_db_service()
        res = face_service.register_customer(customer_id, name, image_bgr)

        return CustomerRegistrationResponse(
            status="success",
            message=res["message"],
            customer_id=res["customer_id"],
            name=res["name"],
            registered_at=res["registered_at"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Customer registration error: {str(e)}",
        )


@router.post(
    "/recognize-customer",
    response_model=CustomerFaceRecognitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Identify facial image against customer database and log visit timestamp",
)
async def recognize_customer(
    file: UploadFile = File(...),
    tolerance: float = Query(
        0.6, ge=0.1, le=1.5, description="Face distance match threshold"
    ),
):
    """Extracts face encodings from query image, matches against face_db.pkl, and logs visit timestamp."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided must be a valid image.",
        )

    try:
        contents = await file.read()
        cv_proc = get_cv_processor()
        image_bgr = cv_proc.load_image(contents)

        face_service = get_face_recognition_db_service()
        res = face_service.identify_customer(image_bgr, tolerance=tolerance)

        return CustomerFaceRecognitionResponse(
            status="success",
            matched=res["matched"],
            customer_id=res.get("customer_id"),
            customer_name=res.get("customer_name"),
            confidence_score=res.get("confidence_score", 0.0),
            distance=res.get("distance"),
            total_visits=res.get("total_visits"),
            last_visit=res.get("last_visit"),
            visit_history=res.get("visit_history"),
            message=res.get("message"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face recognition error: {str(e)}",
        )


@router.get(
    "/customers",
    response_model=List[CustomerProfile],
    status_code=status.HTTP_200_OK,
    summary="Retrieve list of registered customer profiles",
)
async def list_customers():
    """Returns all registered customer records from face_db.pkl."""
    face_service = get_face_recognition_db_service()
    return face_service.get_all_customers()

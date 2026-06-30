from typing import List
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

from app.schemas.predict import PredictionResponse, ModelInfo
from app.services.detector import detector
from app.core.config import settings

router = APIRouter()

async def read_image_file(file: UploadFile) -> np.ndarray:
    """Helper function to read an uploaded image into an OpenCV format array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

@router.get("/model/info", response_model=ModelInfo, tags=["model"])
async def get_model_info():
    """
    Get information about the currently loaded model.
    """
    return ModelInfo(
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        model_path=settings.HF_REPO_ID
    )

@router.post("/predict", response_model=PredictionResponse, tags=["predict"])
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=settings.CONFIDENCE_THRESHOLD)
):
    """
    Run object detection on a single uploaded image.
    """
    img = await read_image_file(file)
    
    try:
        detections, inference_time = detector.predict(img, conf_threshold=confidence_threshold)
        
        return PredictionResponse(
            inference_time_ms=inference_time,
            detections=detections
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=List[PredictionResponse], tags=["predict"])
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(default=settings.CONFIDENCE_THRESHOLD)
):
    """
    Run object detection on multiple uploaded images.
    """
    responses = []
    for file in files:
        img = await read_image_file(file)
        try:
            detections, inference_time = detector.predict(img, conf_threshold=confidence_threshold)
            responses.append(PredictionResponse(
                inference_time_ms=inference_time,
                detections=detections
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
            
    return responses

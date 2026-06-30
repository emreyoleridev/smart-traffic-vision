from typing import List, Tuple
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class DetectionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox

class PredictionResponse(BaseModel):
    inference_time_ms: float
    detections: List[DetectionResult]

class ModelInfo(BaseModel):
    project_name: str
    version: str
    model_path: str

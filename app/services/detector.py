import os
import time
import logging
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.core.config import settings
from app.schemas.predict import DetectionResult, BoundingBox

logger = logging.getLogger(__name__)

class TrafficDetector:
    def __init__(self):
        self.model = None
        self.device = "cpu"  # We can detect GPU here if needed
        self._classes_of_interest = {
            "person", "car", "truck", "bus", "motorcycle", "bicycle", 
            "traffic light", "stop sign"
        }

    def load_model(self):
        """
        Downloads the model from Hugging Face (if not cached) and loads it.
        """
        if self.model is not None:
            return

        logger.info(f"Checking for model from HF: {settings.HF_REPO_ID}")
        
        try:
            # Download model from huggingface
            model_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.HF_MODEL_FILENAME,
                cache_dir="models"
            )
            logger.info(f"Model successfully loaded from {model_path}")
            
            # Load with Ultralytics
            self.model = YOLO(model_path)
            
            # warmup
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_image, verbose=False)
            logger.info("Model warmup complete.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not initialize TrafficDetector: {e}")

    def predict(self, image: np.ndarray, conf_threshold: float = None) -> Tuple[List[DetectionResult], float]:
        """
        Runs inference on the provided image array.
        Returns a tuple of (detections, inference_time_ms)
        """
        if self.model is None:
            self.load_model()
            
        conf = conf_threshold if conf_threshold is not None else settings.CONFIDENCE_THRESHOLD
        
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            source=image, 
            conf=conf, 
            iou=settings.IOU_THRESHOLD,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        detections = []
        
        result = results[0]  # We only passed one image
        
        # BDD100k or COCO classes might be slightly different. 
        # We'll map the predictions dynamically.
        names = result.names
        
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names[class_id].lower()
            confidence = float(box.conf[0].item())
            
            # Optional: Filter only classes of interest
            # if class_name not in self._classes_of_interest:
            #     continue
                
            xyxy = box.xyxy[0].tolist()
            
            detections.append(
                DetectionResult(
                    class_name=class_name,
                    class_id=class_id,
                    confidence=confidence,
                    box=BoundingBox(
                        x_min=xyxy[0],
                        y_min=xyxy[1],
                        x_max=xyxy[2],
                        y_max=xyxy[3]
                    )
                )
            )
            
        return detections, inference_time_ms

    def annotate_image(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Draws bounding boxes and labels on the image.
        """
        annotated = image.copy()
        
        for det in detections:
            # Draw rectangle
            x1, y1, x2, y2 = int(det.box.x_min), int(det.box.y_min), int(det.box.x_max), int(det.box.y_max)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return annotated

# Singleton instance
detector = TrafficDetector()

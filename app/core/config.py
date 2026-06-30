from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Traffic Vision API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Hugging Face Model configuration
    HF_REPO_ID: str = "emreyoleridev/smart-traffic-vision-yolov8"
    HF_MODEL_FILENAME: str = "best.pt"
    
    # We'll save the downloaded model locally here
    MODEL_PATH: str = "models/yolov8_traffic.pt" 
    
    # Inference parameters
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45

    class Config:
        case_sensitive = True

settings = Settings()

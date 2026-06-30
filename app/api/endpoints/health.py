from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["health"])
async def health_check():
    """
    Check if the API is running successfully.
    """
    return {"status": "ok", "message": "API is healthy"}

from fastapi import FastAPI

app = FastAPI(
    title="Smart Traffic Vision API",
    version="1.0.0"
)

@app.get("/")
def health():
    return {
        "status": "running",
        "message": "Smart Traffic Vision API"
    }
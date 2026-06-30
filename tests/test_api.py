from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_health_check():
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy"}

def test_model_info():
    response = client.get(f"{settings.API_V1_STR}/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "project_name" in data
    assert "version" in data

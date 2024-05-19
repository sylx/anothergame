from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World!"}

def test_post_image_generate():
    test_data = {
        "prompt": "a girl",
        "negative_prompt": "low resolution",
        "seed": 42,
        "removeBg": True
    }
    response = client.post("/image/generate", json=test_data)
    assert response.status_code == 200
    assert response.json()["result"] == True
    
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.image import ImageService
import uuid
import os


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str
    seed: int
    removeBg: bool = False

imageService = ImageService()

router = APIRouter()

@router.post("/images/generate")
async def generate_image(request: GenerateImageRequest):
    imageBytes = imageService.generate(request.prompt, request.negative_prompt, request.removeBg, request.seed)
    id = uuid.uuid4()
    filepath = f"storage/images/{id}.webp"
    with open(filepath, "wb") as f:
        f.write(imageBytes)
    return {"path": f"/images/{id}.webp","result": True}

@router.delete("/images/{id}")
async def delete_image(id: str):
    filepath = f"storage/images/{id}.webp"
    try:
        os.remove(filepath)
        return {"result": True}
    except FileNotFoundError:
        return {"result": False}


@router.get("/images/{id}")
async def get_image(id: str):
    filepath = f"storage/images/{id}.webp"
    return StreamingResponse(open(filepath, "rb"), media_type="image/webp")
    
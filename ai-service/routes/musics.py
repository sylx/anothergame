from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.music import MusicService
import uuid
import os
import soundfile as sf

class GenerateMusicRequest(BaseModel):
    prompt: str

musicService = MusicService()

router = APIRouter()

@router.post("/musics/generate")
async def generate_music(request: GenerateMusicRequest):
    audio = musicService.generate(request.prompt)
    id = uuid.uuid4()
    filepath = f"storage/musics/{id}.ogg"
    with open(filepath, "wb") as f:
        sf.write(f, audio, 24000, format='OGG', subtype='OPUS')
    return {"path": f"/musics/{id}.ogg","result": True}

@router.delete("/musics/{id}")
async def delete_music(id: str):
    filepath = f"storage/musics/{id}.ogg"
    try:
        os.remove(filepath)
        return {"result": True}
    except FileNotFoundError:
        return {"result": False}

@router.get("/musics/{id}")
async def get_music(id: str):
    filepath = f"storage/musics/{id}.ogg"
    return StreamingResponse(open(filepath, "rb"), media_type="audio/ogg")

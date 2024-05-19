from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.tts import TtsService
import uuid
import os
import soundfile as sf

class GenerateVoiceRequest(BaseModel):
    text: str
    speaker_id: int = 1

ttsService = TtsService()

router = APIRouter()

@router.post("/voices/generate")
async def generate_voice(request: GenerateVoiceRequest):
    audio = ttsService.generateSpeech(request.text, request.speaker_id)
    id = uuid.uuid4()
    filepath = f"storage/voices/{id}.ogg"
    with open(filepath, "wb") as f:
        sf.write(f, audio, 24000, format='OGG', subtype='OPUS')
    return {"path": f"/voices/{id}.ogg","result": True}

@router.delete("/voices/{id}")
async def delete_voice(id: str):
    filepath = f"storage/voices/{id}.ogg"
    try:
        os.remove(filepath)
        return {"result": True}
    except FileNotFoundError:
        return {"result": False}

@router.get("/voices/{id}")
async def get_voice(id: str):
    filepath = f"storage/voices/{id}.ogg"
    return StreamingResponse(open(filepath, "rb"), media_type="audio/ogg")

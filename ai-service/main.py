from fastapi import FastAPI
from routes import images, voices, musics


app = FastAPI()
app.include_router(images.router)
app.include_router(voices.router)
app.include_router(musics.router)

@app.get("/")
async def root():
    return {"message": "Hello World!"}

import torch
from diffusers import StableDiffusionPipeline
from rembg import remove
import io

import logging

DEVICE = 'cuda'

# モデルダウンロード先ディレクトリ
CACHE_DIR = 'cache'
#MODEL_PATH="/mnt/e/Download/aamAnyloraAnimeMixAnime_v1.safetensors"
MODEL_PATH="models/realisticVisionV60B1_v51HyperVAE.safetensors"


class ImageService:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        logging.info("Loading model")
        self.pipe = StableDiffusionPipeline.from_single_file(MODEL_PATH,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            load_safety_checker=False).to(DEVICE)
        logging.info("Model loaded")

    def generate(self,prompt="",negative_prompt="",removeBg=False,seed=None):
        if self.pipe is None:
            self.load_model()
        logging.info("Generating image")
        image = self.pipe(prompt=prompt, negative_prompt=negative_prompt, seed=seed).images[0]
        logging.info("Image generated")
        if removeBg:
            image = remove(image)
        memFile = io.BytesIO()
        image.save(memFile, format="WEBP")
        return memFile.getvalue()
            
            

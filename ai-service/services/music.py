import numpy as np
import librosa

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

import logging

dev = "cuda:0"

class MusicService():
    def __init__(self) -> None:        
        super().__init__()
        self.model = None
        self.processor = None

    def load_model(self):
        logging.info("Loading model")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small",
            torch_dtype=torch.float16,
            cache_dir="cache"
        ).to(dev)
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small",cache_dir="cache")
        logging.info("Model loaded")

    def generate(self,prompt="rock"):
        if self.model is None:
            self.load_model()
        
        inputs = self.processor(
            text=[
                prompt
                ],
            padding=True,
            return_tensors="pt",
        ).to(dev)
        logging.info("Generating music")
        audio_values = self.model.generate(**inputs, max_new_tokens=1536).to('cpu')
        sampling_rate = self.model.config.sampling_rate
        y=audio_values[0][0].numpy()
        tempo,beats_frame = librosa.beat.beat_track(y=y, sr=sampling_rate)
        if(len(beats_frame)<32):
            return None
        logging.info(f"Music generated with tempo {tempo} and {len(beats_frame)} beats")
        beats_time = librosa.frames_to_time(beats_frame, sr=sampling_rate)
        begin = beats_time[0]
        end = beats_time[32]
        # begin秒からend秒までの音を切り出す
        audio=y[int(begin*sampling_rate):int(end*sampling_rate)]
        audio= audio.astype(np.float32)
        audio = np.ndarray.flatten(audio)
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=24000)
        audio = librosa.util.normalize(audio)
        # 音量を下げる
        audio = audio * 0.8
        return audio


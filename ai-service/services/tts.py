from pathlib import Path

from MoeGoe.MoeGoe import *
import numpy as np
import librosa
import logging

dev = "cuda:0"

from scipy.signal import butter, lfilter

#from libs.tune import autotune

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class TtsService():
    def __init__(self) -> None:        
        super().__init__()
        self.net_g_ms = None

    def load_model(self):
        logging.info("Loading model")
        model = f'models/vits_models/ruise/1158_epochs.pth'
        config = f'models/vits_models/ruise/config.json'
        self.hps_ms = utils.get_hparams_from_file(config)
        n_speakers = self.hps_ms.data.n_speakers if 'n_speakers' in self.hps_ms.data.keys() else 0
        n_symbols = len(self.hps_ms.symbols) if 'symbols' in self.hps_ms.keys() else 0
        self.speakers = self.hps_ms.speakers if 'speakers' in self.hps_ms.keys() else ['0']
        use_f0 = self.hps_ms.data.use_f0 if 'use_f0' in self.hps_ms.data.keys() else False
        emotion_embedding = self.hps_ms.data.emotion_embedding if 'emotion_embedding' in self.hps_ms.data.keys() else False
        self.net_g_ms = SynthesizerTrn(
            n_symbols,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=n_speakers,
            emotion_embedding=emotion_embedding,
            **self.hps_ms.model).to(dev)
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(model, self.net_g_ms)
        logging.info("Model loaded")
    
    def generateSpeech(self,text,speaker_id=1):
        if self.net_g_ms is None:
            self.load_model()

        length_scale, text = get_label_value(text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')
        stn_tst = get_text(text, self.hps_ms, cleaned=cleaned)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(dev)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(dev)
            sid = LongTensor([speaker_id]).to(dev)
            # black magic
            audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            # convert sample rate to 24000 using scipy
            audio = audio.astype(np.float32)
            audio = librosa.resample(audio, orig_sr=22050, target_sr=24000)
            return audio

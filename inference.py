"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/1 13:40
@DESC:
"""

from models.edenTTS import EdenTTS
from hparams import hparams as hp
from utils.paths import Paths
import torch

import time
from utils.dsp import save_wav
from pathlib import Path
import os
from utils.log_util import get_logger
import numpy as np
from text.en_util import text_to_sequence
import argparse

from hifigan import vocoder

device = "cpu"
log = get_logger(__name__)
vocoder.to(device)


def m_inference(tts_model, out_path, texts):
    for text in texts:
        log.info(f"processing text: {text}")
        phones = text_to_sequence(text)
        phones = torch.tensor(phones).long().unsqueeze(0).to(device)
        s1 = time.time()
        mel_pred = tts_model.inference(phones)
        log.info(f"acoustic model inferance time {time.time() - s1}s")
        with torch.no_grad():
            audio = vocoder(mel_pred.transpose(1, 2))
        file = os.path.join(out_path, f'{text[:40]}.wav')
        wav = audio.squeeze().cpu().detach().numpy()
        peak = np.abs(wav).max()
        wav = wav / peak
        save_wav(wav, file)
        log.info(f"Synthesized wave saved at: {file}")


def inference(texts):
    if type(texts) == str:
        texts = [texts]
    tts_model = EdenTTS(hp).to(device)
    tts_model_id = hp.tts_model_id
    paths = Paths(hp.data_path, tts_model_id)
    tts_model_path = paths.tts_latest_weights
    if not os.path.exists(tts_model_path):
        print(f"{tts_model_path} do not exist")
        return
    out_path = paths.tts_output
    os.makedirs(out_path, exist_ok=True)
    tts_model.load(tts_model_path)
    tts_model.to(device)
    m_inference(tts_model, out_path, texts)


if __name__ == "__main__":
    assert hp.speaker == "ljs"
    # set the path to the LJSpeech dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--text", type=str, required=True, help="input text"
    )
    args = parser.parse_args()
    inference(args.text)




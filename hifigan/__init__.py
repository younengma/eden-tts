from .models import Generator
import json
from hparams import hparams as hp
import torch
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


with open("hifigan/config.json", "r") as f:
    config = json.load(f)

config = AttrDict(config)
vocoder = Generator(config)
if os.path.exists(hp.voc_path) is False:
    raise Exception("please download hifigan vocoder and set voc_path in hparams.py")
ckpt = torch.load(hp.voc_path)
vocoder.load_state_dict(ckpt["generator"])
vocoder.eval()
vocoder.remove_weight_norm()
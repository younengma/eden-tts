# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:33:02 2022

monotonic mask

@author: home
"""
import torch
import numpy as np
from hparams import hparams as hp
from utils.paths import Paths
import pickle
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from utils.display import *

paths = Paths(hp.data_path, speaker=hp.speaker)


def generate_W(T1, T2, g=0.2):
    """

    :param T1: number of text tokens
    :param T2: number of mels
    :param g:
    :return:
    """
    n_items = torch.arange(0, T1)/(T1-1)
    t_items = torch.arange(0, T2)/(T2-1)
    w = torch.exp(-(n_items.unsqueeze(1) - t_items.unsqueeze(0))**2/(2*g**2))
    return w


def prepare_energy_weight(item):
    id, mel_len = item
    mel = np.load(paths.mel / f'{id}.npy')
    phone = np.load(paths.phone / f'{id}.npy')
    T1 = len(phone)
    T2 = mel.shape[-1]
    w = generate_W(T1, T2)
    np.save(paths.energy_mask/f'{id}.npy', w, allow_pickle=False)
    return id


if __name__ == "__main__":
    print(f"speaker is {hp.speaker}")
    with open(paths.data / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"number of items in dataset is:{len(dataset)}")
    pool = Pool(processes=6)
    for i, item_id in enumerate(pool.imap_unordered(prepare_energy_weight, dataset), 1):
        bar = progbar(i, len(dataset))
        message = f'{bar} {i}/{len(dataset)} '
        stream(message)

    




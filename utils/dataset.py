import pickle
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from pathlib import Path
import os


###################################################################################
#TTS Dataset ############################################################
###################################################################################


def get_tts_datasets(path: Path, batch_size):
    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    test_file = path / "test_ids.txt"
    test_ids = []
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            for line in f:
                test_ids.append(line.strip())

    val_file = path / "val_ids.txt"
    val_ids = []
    if os.path.exists(val_file):
        with open(test_file, "r") as f:
            for line in f:
                val_ids.append(line.strip())

    train_ids = []
    mel_lengths = []
    longest = 0
    attn_example = None
    for (item_id, _len) in dataset:
        if _len <= hp.tts_max_mel_len:
            if item_id in test_ids or item_id in val_ids:
                continue
            train_ids += [item_id]
            mel_lengths.append(_len)
            if _len > longest:
                longest = _len
                attn_example = item_id

    train_dataset = TTSDataset(path, train_ids)
    sampler = None
    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=Collate_tts(),
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=1,
                           pin_memory=True)
    if len(test_ids) > 0:
        test_dataset = TTSDataset(path, test_ids)
        test_set = DataLoader(test_dataset,
                               collate_fn=Collate_tts(),
                               batch_size=batch_size,
                               sampler=None,
                               num_workers=1,
                               pin_memory=False)
    else:
        test_set = None

    if len(val_ids) > 0:
        val_dataset = TTSDataset(path, val_ids)
        val_set = DataLoader(val_dataset,
                              collate_fn=Collate_tts(),
                              batch_size=batch_size,
                              sampler=None,
                              num_workers=1,
                              pin_memory=False)
    else:
        val_set = None

    return train_set, val_set, test_set, attn_example


# 之所以不用lambda 形式的，是因为在windows上dataloader会报错
class Collate_tts(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        return collate_tts(batch)


class TTSDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, lang="CN"):
        super(TTSDataset, self).__init__()
        self.path = path
        self.metadata = dataset_ids
        self.lang = lang

    def __getitem__(self, index):
        item_id = self.metadata[index]
        mel = np.load(self.path /'mel' / f'{item_id}.npy')
        phone_path = self.path / 'phone' / f'{item_id}.npy'
        phone_ids = np.load(phone_path)
        emask = np.load(self.path / 'mask' / f'{item_id}.npy')
        return phone_ids, mel, emask, item_id

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def pad2ds(x, max_t_len, max_m_len):
    t_len, m_len = x.shape
    return np.pad(x, ((0, max_t_len-t_len), (0, max_m_len - m_len)), mode='constant', constant_values=1)


def collate_tts(batch):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    phones = [pad1d(x[0], max_x_len) for x in batch]
    phones = np.stack(phones)

    mel_lens = [x[1].shape[-1] for x in batch]
    max_mel_len = max(mel_lens)

    mel = [pad2d(x[1], max_mel_len) for x in batch]
    mel = np.stack(mel)

    e_weight = [pad2ds(x[2], max_x_len, max_mel_len) for x in batch]
    e_weight = np.stack(e_weight)

    ids = [x[-1] for x in batch]

    phones = torch.tensor(phones).long()
    text_lens = torch.tensor(x_lens).long()
    mel_lens = torch.tensor(mel_lens).long()
    mel = torch.tensor(mel)
    e_weight = torch.tensor(e_weight).float()
    return phones, text_lens,  mel, mel_lens, e_weight, ids


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)



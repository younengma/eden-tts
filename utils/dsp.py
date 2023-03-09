from scipy.signal import lfilter
import soundfile as sf
import math
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from hparams import hparams as hp
import librosa


def mcd(mel1, mel2, n_mfcc=34):
    mfcc1 = librosa.feature.mfcc(S=mel1, sr=hp.sample_rate, n_mfcc=n_mfcc)
    mfcc2 = librosa.feature.mfcc(S=mel2, sr=hp.sample_rate, n_mfcc=n_mfcc)
    K = 10 / np.log(10) * np.sqrt(2)
    mcd_dist = K * np.mean(np.sqrt(np.sum((mfcc1 - mfcc2) ** 2, axis=1)))
    return mcd_dist


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path):
    return librosa.load(path, sr=hp.sample_rate)[0]


def save_wav(x, path):
    sf.write(path, x.astype(np.float32), samplerate=hp.sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


MAX_WAV_VALUE = 32768.0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def melspectrogram(y, center=False, np=False):
    if np:
        y = torch.from_numpy(y).unsqueeze(0)
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if hp.fmax not in mel_basis:
        # 构造mel_filter banks, 频谱图经过mel系数谱就得到mel谱了, 默认使用slaney, 使用norm
        mel = librosa_mel_fn(hp.sample_rate, hp.n_fft, hp.num_mels, hp.fmin, hp.fmax)
        mel_basis[str(hp.fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(hp.win_length).to(y.device)
    # 音频前后pad若干个0。
    y_ = torch.nn.functional.pad(y.unsqueeze(1), (int((hp.n_fft-hp.hop_length)/2), int((hp.n_fft-hp.hop_length)/2)), mode='reflect')
    y = y_.squeeze(1)
    # 可能为了解决pytorch中mel difference with librosa
    spec = torch.abs(torch.stft(y, hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length,
                                window=hann_window[str(y.device)],
                                center=center, pad_mode='reflect', normalized=False, onesided=True,
                                return_complex=True))
    spec = torch.matmul(mel_basis[str(hp.fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    if np:
        spec = spec.numpy().squeeze(0)
    return spec


def spectrogram(y, center=False, np=False):
    if np:
        y = torch.from_numpy(y).unsqueeze(0)
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if hp.fmax not in mel_basis:
        # 构造mel_filter banks, 频谱图经过mel系数谱就得到mel谱了, 默认使用slaney, 使用norm
        hann_window[str(y.device)] = torch.hann_window(hp.win_length).to(y.device)
    # 音频前后pad若干个0。
    y_ = torch.nn.functional.pad(y.unsqueeze(1), (int((hp.n_fft-hp.hop_length)/2), int((hp.n_fft-hp.hop_length)/2)), mode='reflect')
    y = y_.squeeze(1)
    # 可能为了解决pytorch中mel difference with librosa
    spec = torch.abs(torch.stft(y, hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length,
                                window=hann_window[str(y.device)],
                                center=center, pad_mode='reflect', normalized=False, onesided=True,
                                return_complex=True))
    if np:
        spec = spec.numpy().squeeze(0)
    return spec


def extract_energy(y, np=False):
    if np:
        y = torch.from_numpy(y).unsqueeze(0)
    spec = spectrogram(y) # magnitude
    #energy = torch.sum(spec ** 2, axis=1)
    #energy = torch.sqrt(energy)
    energy = torch.norm(spec, dim=1)
    if np:
        energy = energy.numpy().squeeze()
    return energy




# def spectrogram(y):
#     D = stft(y)
#     # 预测的频率是db_scale 因为人对声音大小的感知为db。
#     # ref_level_db 为20， 20*np.log10(10) = 20 分贝即幅度值为10认为是参考声音。
#     S = amp_to_db(np.abs(D)) - hp.ref_level_db
#     return normalize(S)

def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

'''
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)
min_level_db = -100
ref_level_db = 20

'''

"""
 人耳能感受到的声音的分贝为10-90分贝
 声音normalize -100, 100 之间 映射到0-1之间。
"""
def normalize(S):
    """
    normalize 之后是0到1之间
    Args:
        S:

    Returns:

    """
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def amp_to_db(x):
    """
    最小的幅度值为20*np.log10(1e-5) = -100
    幅度值一般取2e-5, 到20之间即可。
    Args:
        x:

    Returns:

    """
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def pre_emphasis(x):
    return lfilter([1, -hp.preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel [n_mels, n]
    mel spectrogram back into a waveform."""
    amp_mel = np.exp(mel)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=hp.sample_rate,
        n_fft=hp.n_fft, fmin=hp.fmin)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hp.hop_length, win_length=hp.win_length)
    return wav


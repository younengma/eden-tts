import matplotlib as mpl
mpl.use('agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
from utils.log_util import get_logger
import time
import os
from utils.paths import Paths
import json
from pathlib import Path

log = get_logger(__name__)


def get_time_tag():
    return time.strftime("%m%d%H%M")


def save_stats(stats: dict, paths: Paths, step):
    file = paths.tts_checkpoints/"stats.txt"
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats)+"\n")


def stats_str(stats: dict):
    v = ""
    for key, value in stats.items():
        if value < 0:
            continue
        v += f"{key}:{'%.4f'%value} "
    return v


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


def simple_table(item_tuples):

    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples:

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)):

        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')
    # log.info(f"\n{border}\n{head}\n{body}\n{border}\n")


def time_since(started):
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60:
        h = int(m // 60)
        m = m % 60
        return f'{h}h {m}m {s}s'
    else:
        return f'{m}m {s}s'


def save_attention(attn, path):
    if type(path) == str:
        path = Path(path)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(path.parent/f'{path.stem}.png', bbox_inches='tight')
    plt.close(fig)


def save_spectrogram(M, path, length=None):
    M = np.flip(M, axis=0)
    if length: M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)


def save_wavfig(wav, file):
    plt.plot(wav)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    if file.endswith(".png"):
        plt.savefig(file)
    else:
        plt.savefig(file+".png")
    plt.close()


def plot(array):
    mpl.interactive(True)
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis='x', colors='grey', labelsize=23)
    ax.tick_params(axis='y', colors='grey', labelsize=23)
    plt.plot(array)
    mpl.interactive(False)


def plot_spec(M):
    mpl.interactive(True)
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18,4))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.show()
    mpl.interactive(False)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_mel(data, path, basename, titles=None):
    """

    Args:
        data: [mel, mel, mel] # mel list [80, n_frames]
        path: save path
        basename:  save path

    Returns:

    """
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    plt.savefig(os.path.join(path, "{}.png".format(basename)))
    plt.close()




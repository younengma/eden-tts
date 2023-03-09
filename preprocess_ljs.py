"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/1 10:00
@DESC: 

"""
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
from text.en_util import text_to_sequence
from utils.files import get_files
from pathlib import Path
import argparse
from energy_weight_processor import prepare_energy_weight


paths = Paths(hp.data_path, speaker="ljs")


def convert_file(path: Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    y /= peak
    mel = melspectrogram(y, np=True)
    quant = float_2_label(y, bits=16)
    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path: Path):
    wav_id = path.stem
    m, x = convert_file(path)
    np.save(paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]


def main(wav_path, n_workers=4):
    # 0. wav processing
    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}')
    ])
    print("wav processing........")
    wav_files = get_files(wav_path, ".wav")
    pool = Pool(processes=n_workers)
    dataset = []
    for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        dataset += [(item_id, length)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)
    with open(paths.data / 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    # 1. phone processing
    log.info(f"data path is {paths.data}, token type:{hp.token_type}")
    print("wav processing........")
    meta_file = get_files(wav_path, ".csv")[0]
    with open(meta_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            items = line.split("|")
            id = items[0]
            text = items[-1]
            if hp.token_type == "char":
                phone = np.array(text_to_sequence(text, token_type="char"))
            else:
                phone = np.array(text_to_sequence(text, token_type="ph"))
            np.save(paths.phone / f"{id}.npy", phone, allow_pickle=False)
            bar = progbar(i, len(lines))
            message = f'{bar} {i}/{len(lines)} '
            stream(message)
    pool.close()

    # 2. energy_weight prepare
    print("prepare energy weight........")
    pool = Pool(processes=n_workers)
    for i, item_id in enumerate(pool.imap_unordered(prepare_energy_weight, dataset), 1):
        bar = progbar(i, len(dataset))
        message = f'{bar} {i}/{len(dataset)} '
        stream(message)
    pool.close()
    print(f'\n\nCompleted. total number of training items is {len(dataset)} \n')


if __name__ == "__main__":
    # make sure the speaker is "ljs" if dataset "LJSpeech"
    assert hp.speaker == "ljs"
    # set the path to the LJSpeech dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--wav_path", type=str, required=False, help="path to ljspeech dataset"
    )
    parser.add_argument("-n", "--n_workers", type=int, default=4)
    args = parser.parse_args()
    args.wav_path = r"G:\dataset\LJSpeech\LJSpeech-1.1"
    main(args.wav_path, n_workers=args.n_workers)


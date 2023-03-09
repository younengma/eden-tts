from torch import optim
from utils.display import *
from utils.dataset import get_tts_datasets
from torch.utils.data import DataLoader
from utils.paths import Paths
from models.edenTTS import EdenTTS
import time
from utils.display import stats_str, save_stats
from utils.checkpoints import save_checkpoint, restore_checkpoint
from utils.log_util import get_logger
from models.loss import *
from hparams import hparams as hp

log = get_logger(__name__)


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


torch.autograd.set_detect_anomaly(True)


def main():
    # Parse Arguments
    paths = Paths(hp.data_path, hp.tts_model_id, speaker=hp.speaker)
    device = torch.device('cuda')
    log.info(f"train :{hp.tts_model_id}, checkpoint path:{paths.tts_checkpoints}, batch_size:{hp.batch_size}, Using device:{device}")
    batch_size = hp.batch_size
    training_steps = hp.tts_max_steps
    lr = hp.lr
    # Instantiate Tacotron Model
    log.info('\nInitialising  Model...\n')

    model = EdenTTS(hp).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)
    train_set, val_set, test_set, attn_example = get_tts_datasets(paths.data, batch_size)
    log.info(f"atten exmaple is: {attn_example}")
    tts_train(paths, model, optimizer, train_set, lr, training_steps, attn_example)
    log.info('\n\n training completed!')


def tts_train(paths: Paths, model: EdenTTS, optimizer, train_set: DataLoader, lr, train_steps, attn_example):

    device = next(model.parameters()).device  # use same device as model parameters
    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):
        start = time.time()
        running_loss = 0
        ave_mel_loss = 0
        ave_mel_loss_post = 0
        for i, batch in enumerate(train_set, 1):
            phones, text_lens, m, mel_lens, e_mask = [v.to(device) for v in batch[:-1]]
            ids = batch[-1]
            # Parallelize model onto GPUS using workaround due to python bug， 并行训练速度并不快
            outputs = model(phones, text_lens, m, mel_lens, e_mask)
            step = model.get_step()
            mel_pred_post = None
            if len(outputs) == 5:
                log_dur_pred, log_dur_target, mel_pred, atten0, atten1 = outputs
            else:
                log_dur_pred, log_dur_target, mel_pred, mel_pred_post, atten0, atten1 = outputs
            m = m.transpose(1, 2)
            mel_loss = mel_loss_func(mel_pred, m, mel_lens)
            dur_loss = duration_loss_func(log_dur_pred, log_dur_target, text_lens)
            loss = dur_loss + mel_loss
            if mel_pred_post is not None:
                mel_loss_post = mel_loss_func(mel_pred_post, m, mel_lens)
                loss = loss + mel_loss_post
            else:
                mel_loss_post = None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats = dict()
            if mel_loss is not None:
                ave_mel_loss += mel_loss.item()
                stats["avemel"] = ave_mel_loss / i
                stats["mel"] = mel_loss.item()
            if mel_loss_post is not None:
                ave_mel_loss_post += mel_loss_post.item()
                stats["avemel_post"] = ave_mel_loss_post/i
                stats["mel_post"] = mel_loss_post.item()
            if dur_loss is not None:
                stats["dur"] = dur_loss.item()
            running_loss += loss.item()
            stats["aveloss"] = running_loss / i
            stats["step"] = step
            stats["loss"] = loss.item()
            speed = i / (time.time() - start)
            save_stats(stats, paths, step)
            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'step_{step}'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)
            if attn_example in ids:
                idx = ids.index(attn_example)
                mel_len = mel_lens[idx].item()
                p_mel = mel_pred[idx, :mel_len, :]
                save_spectrogram(np_now(p_mel), paths.tts_mel_plot / f'{step}', 600)
                if attn_example in ids:
                    idx = ids.index(attn_example)
                    atten = atten0[idx]
                    text_len = text_lens[idx].item()
                    save_attention(np_now(atten[:text_len, :mel_len]), paths.tts_attention / f'{step}s')
                    atten = atten1[idx]
                    text_len = text_lens[idx].item()
                    mel_len = mel_lens[idx].item()
                    save_attention(np_now(atten[:text_len, :mel_len]), paths.tts_attention / f'{step}h')

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters})|{stats_str(stats)} {speed:#.2} steps/s'
            if step % hp.tts_show_info_every == 0:
                log.info(msg)

            if step >= train_steps:
                save_checkpoint('tts', paths, model, optimizer, is_silent=False)
                break

        # Must save latest optimizer state to ensure that resuming training
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        with open(paths.tts_checkpoints / "epoch_loss.txt", 'a', encoding='utf-8') as fhand:
            line = f'epoch:{e}\tepoch_loss:{running_loss}\tstep:{step}\n'
            fhand.write(line)


if __name__ == "__main__":
    main()

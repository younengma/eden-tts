import torch
from utils.tools import get_mask_from_lengths
import torch.nn.functional as F


def format_Wnt(N, T, g=0.2):
    n_items = torch.arange(0, N, device=N.device)/N
    t_items = torch.arange(0, T, device=N.device)/T
    w = 1 - torch.exp(-(n_items.unsqueeze(1) - t_items.unsqueeze(0))**2/(2*g**2))
    # 中间的值weighth很小，让loss不会注意到他们，非中间的值weight较大
    return w


def guided_atten_loss_func(a, text_lens, mel_lens):
    loss = 0.0
    for i, (N, T) in enumerate(zip(text_lens, mel_lens)):
        w = format_Wnt(N, T).to(a.device)
        loss += torch.sum(a[i, :N, :T]*w)
    return loss/sum(mel_lens)


def duration_loss_func(d_pred, d_target, ilens):
    duration_masks = ~get_mask_from_lengths(ilens).to(d_pred.device)
    d_outs = d_pred.masked_select(duration_masks)
    ds = d_target.masked_select(duration_masks)
    return F.l1_loss(d_outs, ds)


def pe_loss_func(d_pred, d_target, ilens):
    duration_masks = ~get_mask_from_lengths(ilens).to(d_pred.device)
    d_target.requires_grad = False
    d_outs = d_pred.masked_select(duration_masks)
    ds = d_target.masked_select(duration_masks)
    return F.mse_loss(d_outs, ds)


def mel_loss_func(mel_pred, mel_target, mel_lens):
    """
    mel_pred: B, L, n_mels
    """
    mel_masks = ~get_mask_from_lengths(mel_lens, max_len=mel_pred.shape[1]).to(mel_pred.device)
    mel_target = mel_target[:, :mel_masks.shape[1], :]
    mel_target.requires_grad = False
    mel_target = mel_target.masked_select(mel_masks.unsqueeze(-1))
    mel_pred = mel_pred.masked_select(mel_masks.unsqueeze(-1))
    return F.mse_loss(mel_pred, mel_target)

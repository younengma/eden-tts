"""
@author: edenmyn
@email: edenmyn
@time: 2022/7/29 8:49
@DESC: 

"""
import torch


def format_Wnt(N, T, g=0.2):
    n_items = torch.arange(0, N, device=N.device)/N
    t_items = torch.arange(0, T, device=N.device)/T
    w = 1 - torch.exp(-(n_items.unsqueeze(1) - t_items.unsqueeze(0))**2/(2*g**2))
    # 中间的值weighth很小，让loss不会注意到他们，非中间的值weight较大
    return w


def guided_atten_loss(a, text_lens, mel_lens):
    loss = 0.0
    for i, (N, T) in enumerate(zip(text_lens, mel_lens)):
        w = format_Wnt(N, T).to(a.device)
        loss += torch.sum(a[i, :N, :T]*w)
    return loss/sum(mel_lens)


def print_loss(stats:dict):
    res = ""
    for key, value in stats.items():
        res += "%s:%.5f "%(key, value)
    return res

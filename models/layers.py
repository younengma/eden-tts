"""
@author: edenmyn
@email: edenmyn
@time: 2022/7/8 14:37
@DESC:

"""
import torch
from utils.log_util import get_logger
from utils.net_utils import get_padding
import numpy as np
from utils.tools import get_mask_from_lengths
from hparams import hparams as hp

logging = get_logger(__name__)


def reconstruct_align_from_aligned_position(
             e, delta=0.2, mel_lens=None,
            text_lens=None, max_mel_len=None):
    """Reconstruct alignment matrix from aligned positions.
            Args:
                e: aligned positions [B, T1].
                delta: a scalar, default 0.01
                mel_mask: mask of mel-spectrogram [B, T2], None if inference and B==1.
                text_mask: mask of text-sequence, None if B==1.
            Returns:
                alignment matrix [B, T1, T2].
            """
    b, T1 = e.shape
    if mel_lens is None:
        assert b == 1
        max_length = torch.round(e[:, -1]).squeeze().item()
    else:
        if max_mel_len is None:
            max_length = mel_lens.max()
        else:
            max_length = max_mel_len

    q = torch.arange(0, max_length).unsqueeze(0).repeat(e.size(0), 1).to(e.device).float()
    if mel_lens is not None:
        mel_mask = get_mask_from_lengths(mel_lens, max_len=max_length).to(e.device)
        q = q * (~mel_mask).float()
    energies = -1 * delta * (q.unsqueeze(1) - e.unsqueeze(-1)) ** 2
    if text_lens is not None:
        text_mask = get_mask_from_lengths(text_lens, max_len=T1).to(e.device)
        energies = energies.masked_fill(
            text_mask.unsqueeze(-1).repeat(1, 1, max_length),
            -float('inf')
        )

    alpha = torch.softmax(energies, dim=1)
    if mel_lens is not None:
        alpha = alpha.masked_fill(
            mel_mask.unsqueeze(1).repeat(1, text_mask.size(1), 1),
            0.0
        )

    return alpha


def scaled_dot_attention(key, key_lens, query, query_lens, e_weight=None):
    dim = key.size(-1)
    T1 = query.size(1)
    N1 = key.size(1)
    device = key.device
    energies = query @ key.transpose(1, 2) / np.sqrt(float(dim))
    if e_weight is not None:
        energies = energies*e_weight.transpose(1, 2)

    key_mask = get_mask_from_lengths(key_lens, max_len=N1).to(device)
    key_mask = key_mask.unsqueeze(1).repeat(1, T1, 1)
    energies = energies.masked_fill(key_mask, -float("inf"))
    alpha = torch.softmax(energies, dim=-1)

    query_mask = get_mask_from_lengths(query_lens, max_len=T1).to(device)
    query_mask = query_mask.unsqueeze(2).repeat(1, 1, N1)
    alpha = alpha.masked_fill(query_mask, 0.0)
    return alpha.transpose(1, 2)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout: output dim size
        dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): input tensor
        Returns:
            layer normalized tensor
        """
        if self.dim == -1:
            return super().forward(x)
        else:
            return super().forward(x.transpose(1, -1)).transpose(1, -1)


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ResConv1d(torch.nn.Module):
    """Residual Conv1d layer"""

    def __init__(
            self,
            n_channels=512,
            k_size=5,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.1},
            dropout_rate=0.1,
            dilation=1
    ):
        super().__init__()
        if dropout_rate < 1e-5:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels, n_channels,
                    kernel_size=k_size, padding=(k_size - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels, n_channels,
                    kernel_size=k_size, padding=get_padding(kernel_size=k_size, dilation=dilation),
                    dilation=dilation
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        # x [B, C, T]
        x = x + self.conv(x)
        return x


class ResConvBlock(torch.nn.Module):
    """Block containing several ResConv1d layers."""

    def __init__(
            self,
            num_layers,
            n_channels=512,
            k_size=5,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.1},
            dropout_rate=0.1,
            use_weight_norm=True,
            dilations=None
    ):
        super().__init__()
        self.num_layers = num_layers
        if dilations is not None:
            blocks = []
            for i, dialation in enumerate(dilations):
                blocks.append(ResConv1d(n_channels, k_size,
                                        nonlinear_activation,
                                        nonlinear_activation_params,
                                        dropout_rate, dialation))
            self.layers = torch.nn.Sequential(*blocks)
        else:
            self.layers = torch.nn.Sequential(*[
                ResConv1d(n_channels, k_size, nonlinear_activation,
                          nonlinear_activation_params, dropout_rate) \
                for _ in range(num_layers)
            ])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        # x: [B, C, T]
        return self.layers(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class TokenEmbedding(torch.nn.Module):
    def __init__(self, hidden_size=384, padding_idx=0, vocab_size=365):
        super().__init__()
        self.phone_embed_layer = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, phone_ids):
        phone_embeddings = self.phone_embed_layer(phone_ids)
        return phone_embeddings
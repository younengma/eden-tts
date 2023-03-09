"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/16 10:54
@DESC: 

"""
from .layers import *
import torch.nn as nn
from transformer.Layers import FFTBlock
from models.layers import TokenEmbedding
from transformer.Models import get_sinusoid_encoding_table
from collections import OrderedDict
from models.modules import Conv, Linear


class TextEncoder(torch.nn.Module):
    """
     this is the text encoder adapted from fastspeech
     we add make a small modification on the position embedding
    """
    def __init__(self, encoder_layer=5,
        encoder_head =2,
        encoder_hidden=256,
        conv_filter_size=1024,
        conv_kernel_size=[9, 1],
        encoder_dropout=0.2,
        n_channels=512,
        vocab_size=365):
        super().__init__()
        max_seq_len = 1000
        n_position = max_seq_len + 1
        d_word_vec = n_channels
        n_layers = encoder_layer
        n_head = encoder_head
        d_k = d_v = (
                encoder_hidden
                // encoder_head
        )
        d_model = encoder_hidden
        d_inner = conv_filter_size
        kernel_size = conv_kernel_size
        dropout = encoder_dropout

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.src_word_emb = TokenEmbedding(hidden_size=d_word_vec, padding_idx=0, vocab_size=vocab_size)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.pre_linear = torch.nn.Linear(in_features=n_channels, out_features=encoder_hidden)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.linear_key = nn.Linear(encoder_hidden, n_channels)
        self.linear_value = nn.Linear(encoder_hidden, n_channels)

    def forward(self, src_seq, text_lengths, return_attns=False):
        mask = get_mask_from_lengths(text_lengths).to(src_seq.device)
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # use abosolute positonal embedding, same as the original fastspeech FFT
        if hp.pos_embed_scheme == "absolute":
            # -- Forward
            if not self.training and src_seq.shape[1] > self.max_seq_len:
                enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                    src_seq.shape[1], self.d_model
                )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                    src_seq.device
                )
            else:
                enc_output = self.src_word_emb(src_seq) + self.position_enc[
                    :, :max_len, :
                ].expand(batch_size, -1, -1)
        else:
        # do not use the position embedding or use relative position embedding
        # leads to better performance especially for longer sentences
            enc_output = self.src_word_emb(src_seq)
        enc_output = self.pre_linear(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        text_key = self.linear_key(enc_output)
        text_value = self.linear_value(enc_output)
        return text_key, text_value

    def inference(self, phone_ids: torch.Tensor):
        text_lens = torch.Tensor([phone_ids.size(1)]).long()
        text_key, text_value = self.forward(phone_ids, text_lens)
        return text_value


class MelEncoder(torch.nn.Module):
    def __init__(self, n_mels,
                 n_channels,
                 nonlinear_activation,
                 nonlinear_activation_params,
                 dropout_rate,
                 n_mel_encoder_layer,
                 k_size,
                 use_weight_norm,
                 dilations=None
                 ):
        super().__init__()
        self.mel_prenet = torch.nn.Sequential(
            torch.nn.Linear(n_mels, n_channels),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            torch.nn.Dropout(dropout_rate),
        )
        self.mel_encoder = ResConvBlock(
            num_layers=n_mel_encoder_layer,
            n_channels=n_channels,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
            dilations=dilations
        )

    def forward(self, speech):
        mel_h = self.mel_prenet(speech).transpose(1, 2)
        mel_h = self.mel_encoder(mel_h).transpose(1, 2)
        return mel_h


class Decocer(torch.nn.Module):
    def __init__(self,
                 idim,
                 encoder_hidden,
                 n_decoder_layer,
                 k_size,
                 nonlinear_activation,
                 nonlinear_activation_params,
                 dropout_rate,
                 use_weight_norm,
                 n_mels,
                 dialations=None
                 ):
        super().__init__()
        self.pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.decoder = ResConvBlock(
            num_layers=n_decoder_layer,
            n_channels=encoder_hidden,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
            dilations=dialations
        )
        self.mel_output_layer = torch.nn.Linear(encoder_hidden, n_mels)

    def forward(self, text_value_expanded):
        x = self.pre_linear(text_value_expanded.transpose(1, 2))
        mel_pred = self.decoder(x.transpose(1, 2))
        mel_pred = self.mel_output_layer(mel_pred.transpose(1, 2))
        return mel_pred


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, idim, filter_size=256, ksize=3, dropout=0.1, offset=1):
        super(DurationPredictor, self).__init__()
        self.input_size = idim
        self.filter_size = filter_size
        self.kernel = ksize
        self.conv_output_size = filter_size
        self.dropout = dropout
        self.offset = offset

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        # predict log(d_target + offset)
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        return out

    def inference(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        out = torch.clamp(out.exp() - self.offset, min=0)
        return out

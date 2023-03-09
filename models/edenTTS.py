#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# edenmyn 2022/09/22

from typing import Dict
from typing import Tuple
from models.abtract_model import AbstractModel
from .components import *
from utils.net_utils import parameter_count, make_non_pad_mask
from models.modules import PostNet
from hparams import Hparams

logging = get_logger(__name__)


class EdenTTS(AbstractModel):
    def __init__(
        self,
        h: Hparams,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        duration_offset=1.0,
    ):
        super().__init__()
        self.duration_offset = duration_offset
        self.delta = h.delta

        self.text_encoder = TextEncoder(n_channels=h.n_channels,
                                        encoder_layer=h.text_encoder_layers,
                                        encoder_hidden=h.text_encoder_hidden,
                                        encoder_dropout=h.text_encoder_dropout,
                                        vocab_size=h.vocab_size)

        self.mel_encoder = MelEncoder(n_mels=h.num_mels,
                                      n_channels=h.n_channels,
                                      nonlinear_activation=nonlinear_activation,
                                      nonlinear_activation_params=nonlinear_activation_params,
                                      dropout_rate=h.mel_encoder_dropout,
                                      n_mel_encoder_layer=h.mel_encoder_layers,
                                      k_size=h.mel_encoder_ksize,
                                      use_weight_norm=use_weight_norm,
                                      dilations=h.mel_encoder_dilation)

        self.duration_predictor = DurationPredictor(
            idim=h.n_channels,
            filter_size=h.duration_predictor_filter_zie,
            ksize=h.duration_predictor_ksize,
            dropout=h.duration_predicotr_dropout
        )

        self.decoder = Decocer(idim=h.n_channels,
                               encoder_hidden=h.decoder_hidden,
                               n_decoder_layer=h.decoder_layers,
                               k_size=h.decoder_ksize,
                               nonlinear_activation=nonlinear_activation,
                               nonlinear_activation_params=nonlinear_activation_params,
                               dropout_rate=h.decoder_dropout,
                               use_weight_norm=use_weight_norm,
                               n_mels=h.num_mels,
                               dialations=h.decoder_dilation)

        if h.use_postnet:
            self.postnet = PostNet()
        else:
            self.postnet = None
        te = parameter_count(self.text_encoder)
        de = parameter_count(self.decoder)
        du = parameter_count(self.duration_predictor)
        logging.info(f"tol_params: {parameter_count(self)}, "
                     f"text_encoder:{te},"
                     f"decoder:{de},"
                     f"dur:{du},"
                     f"tol_infer:{te + de + du}")
        
    def forward(
        self,
        phone_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        mel_lens: torch.Tensor,
        e_weight=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward propagations.
        Args:
            text: Batch of padded text ids (B, T1).
            text_lengths: Batch of lengths of each input batch (B,).
            speech: Batch of mel-spectrograms (B, T2, num_mels)
            mel_lens: Batch of mel-spectrogram lengths (B,)
            e_weight: energy weight (B, T1, T2)
        """
        if self.training:
            self.step += 1
        device = phone_ids.device
        mel_mask = ~make_non_pad_mask(mel_lens).to(device)
        text_key, text_value = self.text_encoder(phone_ids, text_lengths)
        speech = speech.transpose(1, 2)
        mel_h = self.mel_encoder(speech)

        alpha = scaled_dot_attention(key=text_key, key_lens=text_lengths, query=mel_h,
                                     query_lens=mel_lens, e_weight=e_weight)
        dur0 = torch.sum(alpha, dim=-1)
        e = torch.cumsum(dur0, dim=-1)
        e = e - dur0/2
        reconst_alpha = reconstruct_align_from_aligned_position(e, mel_lens=mel_lens,
                                                                text_lens=text_lengths,
                                                                delta=self.delta)

        log_dur_pred = self.duration_predictor(text_value)
        log_dur_target = torch.log(dur0.detach() + self.duration_offset)

        text_value_expanded = torch.bmm(
            text_value.transpose(1, 2), reconst_alpha
        )
        _tmp_mask_2 = mel_mask.unsqueeze(1).repeat(1, text_value.size(2), 1)
        text_value_expanded = text_value_expanded.masked_fill(_tmp_mask_2, 0.0)
        mel_pred = self.decoder(text_value_expanded)
        if self.postnet is not None:
            mel_pred_post = self.postnet(mel_pred)
            return log_dur_pred, log_dur_target, mel_pred, mel_pred_post, alpha, reconst_alpha
        else:
            return log_dur_pred, log_dur_target, mel_pred, alpha, reconst_alpha

    def inference(
        self,
        phone_ids: torch.Tensor,
        delta=None,
        d_control=1
    ):
        """Inference.
        Args:
            phone_ids: Batch of padded text ids (1, T1).
            d_control: duration adjust parameter to control synthesis speed
            delta: hperparmeter usd to reconstuct monotonic alignment from durations
        """
        if delta is None:
            delta = self.delta
        self.eval()
        with torch.no_grad():
            text_value = self.text_encoder.inference(phone_ids)
            dur = self.duration_predictor.inference(text_value)*d_control
            if torch.sum(dur)/dur.size(0) < 1:
                dur = 4*torch.ones_like(dur)
                logging.warn("predict too short duration, use dummy ones")
            e = torch.cumsum(dur, dim=1) - dur/2
            alpha = reconstruct_align_from_aligned_position(e, mel_lens=None, text_lens=None, delta=delta)
            text_value_expanded = torch.bmm(
                text_value.transpose(1, 2), alpha
            )
            mel_pred = self.decoder(text_value_expanded)
            if self.postnet is not None:
                mel_pred = self.postnet(mel_pred)
        self.train()
        return mel_pred


import os
import json


# Global settings
class Hparams(object):
    def __init__(self, data=None):
        # this is a configuration file, the setting in the file
        # will overwrite default settings in this object
        # the name of the config file will be used as experimental id
        self.config_file = f"config/eden.json"
        if data is not None:
            self.config_file = data
        print(f"config_file: {self.config_file}")
        # path to the pretrained hifigan vocoder
        # pretrained hifigan vocoder can be downloaded from: https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y
        # we use the model LJ_FT_T2_V1 in the article
        self.voc_path = "path/to/hifigan_vocoder"
        self.voc_path = r"C:\Users\home\Desktop\hifigan\pretrained\lj_wo_ft.pth.tar"
        self.vocab_size = 365  # number of input tokens
        # Training
        self.tts_max_steps = 300_000  # you may stop at around 100_000 for ljspeech dataset for acceptable speech quality
        self.tts_max_mel_len = 1500
        self.tts_bin_lengths = True   # bins the spectrogram lengths before sampling in data loader - speeds up training
        self.tts_checkpoint_every = 400_00  # checkpoints the model every X steps
        self.tts_show_info_every = 2   # print tran status every X steps
        self.tts_eval_every = 30
        self.lr = 1e-4
        self.batch_size = 96

        # token type, can be char or phonemes,
        # we use char in our article
        self.token_type = "char"
        # wether to add a postnet after the decoder
        self.use_postnet = False
        # the speaker aims to specify the task space of the training. You can include other dataset in your datapath.
        self.speaker = "ljs"

        self.base_path = r'./'
        self.data_path = os.path.join("./", r'data')

        # either "none" or "absolute", using the absolute position will lead to the original FFT of fastspeech
        # in our experiment we found using none position position embedding in our architecture
        # lead to slightly better results, especially for long sentences
        self.pos_embed_scheme = "none"  # ["none", "absolute"]
        # CONFIG -------------------------------------------------------------------------------------------------#
        # model configs, will be overiten by the config file
        self.delta = 0.2
        self.n_channels = 512
        self.text_encoder = "fft"
        self.text_encoder_layers = 5
        self.text_encoder_dropout = 0.2
        self.text_encoder_hidden = 384

        self.decoder_dropout = 0.2
        self.decoder_layers = 6
        self.decoder_ksize = 5
        self.decoder_dilation = [1, 2, 2, 2, 1, 1]
        self.decoder_hidden = 512

        self.mel_encoder_layers = 4
        self.mel_encoder_dilation = [1, 2, 2, 3]
        self.mel_encoder_ksize = 5
        self.mel_encoder_dropout = 0.1
        self.mel_encoder_hidden = 512

        self.duration_predictor_ksize = 3
        self.duration_predictor_filter_zie = 256
        self.duration_predicotr_dropout = 0.5

        # DSP --------------------------------------------------------------------------------------------------------#
        # Settings for all models fix
        self.sample_rate = 22050
        self.n_fft = 1024
        self.fft_bins = self.n_fft // 2 + 1
        self.num_mels = 80
        self.hop_length = 256 # 12.5ms - in line with Tacotron 2 paper
        self.win_length = 1024   # 50ms - same reason as above
        self.fmin = 0
        self.fmax = 8000
        self.bits = 16  # bit depth of signal

        # ----------------------------------------------------------------------------------------------------------------#
        # overwrite default settings with config file
        if data is not None:
            self.config_file = data
        if os.path.exists(self.config_file):
            from pathlib import Path
            data = self.config_file
            tag = Path(data).stem
            with open(data, 'r', encoding="utf-8") as fhand:
                data = json.load(fhand)
            for key, value in data.items():
                setattr(self, key, value)
        else:
            raise Exception("empty config file")
        self.tts_model_id = f"tts_{tag}"


hparams = Hparams()

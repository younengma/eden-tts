import os
from pathlib import Path
from hparams import hparams as hp


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path, tts_id=None, speaker=None):
        self.base = Path(hp.base_path)
        self.speaker = speaker
        self.tts_id = tts_id
        # Data Paths
        if speaker is not None:
            self.data = Path(data_path).expanduser().resolve()/speaker
            self.quant = self.data/'quant'
            self.mel = self.data/'mel'
            self.gta = self.data/'gta'
            self.phone = self.data/'phone'
            self.energy_mask = self.data/'mask'
            self.dur = self.data/'dur'
            self.tts_id = tts_id
            self.pitch = self.data/'pitch'
            self.energy = self.data/'energy'
        if tts_id is not None:
            # Tactron/TTS Paths
            self.tts_checkpoints = self.base/'checkpoints'/hp.speaker/f'{tts_id}'
            self.tts_latest_weights = self.tts_checkpoints/'latest_weights.pyt'
            self.voc_latest_weights = self.tts_checkpoints / 'g_latest'
            self.tts_latest_optim = self.tts_checkpoints/'latest_optim.pyt'
            self.tts_output = self.base/'outputs'/hp.speaker/f'{tts_id}'
            self.voc_output = self.base/'outputs'/hp.speaker/f'{tts_id}'
            self.tts_step = self.tts_checkpoints/'step.npy'
            self.tts_log = self.tts_checkpoints/'log.txt'
            self.tts_attention = self.tts_checkpoints/'attention'
            self.tts_mel_plot = self.tts_checkpoints/'mel_plots'
        self.create_paths()

    def create_paths(self):
        if self.speaker is not None:
            os.makedirs(self.data, exist_ok=True)
            os.makedirs(self.quant, exist_ok=True)
            os.makedirs(self.mel, exist_ok=True)
            os.makedirs(self.gta, exist_ok=True)
            os.makedirs(self.phone, exist_ok=True)
            os.makedirs(self.energy_mask, exist_ok=True)
            os.makedirs(self.dur, exist_ok=True)
            os.makedirs(self.energy, exist_ok=True)
            os.makedirs(self.pitch, exist_ok=True)
        if self.tts_id is not None:
            os.makedirs(self.tts_checkpoints, exist_ok=True)
            os.makedirs(self.tts_output, exist_ok=True)
            os.makedirs(self.tts_attention, exist_ok=True)
            os.makedirs(self.tts_mel_plot, exist_ok=True)

    def get_tts_named_weights(self, name):
        """Gets the path for the weights in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_weights.pyt'

    def get_tts_named_optim(self, name):
        """Gets the path for the optimizer state in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_optim.pyt'

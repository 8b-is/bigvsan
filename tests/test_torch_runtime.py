import json
from pathlib import Path
import unittest

import torch

from env import AttrDict
from meldataset import mel_spectrogram
from models import BigVSAN


class TorchRuntimeTest(unittest.TestCase):
    def test_generator_and_mel_preprocessing(self):
        config = Path(__file__).resolve().parents[1] / "configs/bigvsan_base_24khz_100band.json"
        with config.open() as stream:
            settings = AttrDict(json.load(stream))
        settings.upsample_initial_channel = 32
        model = BigVSAN(settings).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, settings.num_mels, 4))
            mel = mel_spectrogram(
                torch.zeros(1, 8192), settings.n_fft, settings.num_mels,
                settings.sampling_rate, settings.hop_size, settings.win_size,
                settings.fmin, settings.fmax,
            )
        self.assertEqual(tuple(output.shape), (1, 1, 1024))
        self.assertEqual(tuple(mel.shape), (1, 100, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(mel).all())

# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np
import config


class Audio():
    def __init__(self):
        self.mel_basis = librosa.filters.mel(sr=config.audio['sample_rate'],
                                             n_fft=config.embedder['n_fft'],
                                             n_mels=config.embedder['num_mels'])

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=config.embedder['n_fft'],
                              hop_length=config.audio['hop_length'],
                              win_length=config.audio['win_length'],
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - config.audio['ref_level_db']
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + config.audio['ref_level_db'])
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=config.audio['n_fft'],
                            hop_length=config.audio['hop_length'],
                            win_length=config.audio['win_length'])

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j*phase)
        return librosa.istft(stft_matrix,
                             hop_length=config.audio['hop_length'],
                             win_length=config.audio['win_length'])

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -config.audio['min_level_db'], -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -config.audio['min_level_db']

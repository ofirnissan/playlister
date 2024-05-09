import os
import librosa
from audio_hw2_code_only.plotter import Plotter


class Song:
    def __init__(self, filepath, epsilon=0, remove_zero_amp=True, sr=48000):
        self.song_name = filepath.split(os.sep)[-1].split('.')
        self.sr = None
        self.audio = None

        self._load_audio(filepath, sr=sr)
        self.plotter = Plotter(self.audio, self.sr)

        if remove_zero_amp:
            self.remove_zero_amplitude(epsilon=epsilon)

    def _load_audio(self, filepath, sr=48000):
        audio, sr = librosa.load(filepath, sr=sr)
        self.audio = audio
        self.sr = sr

    def remove_zero_amplitude(self, epsilon=0):
        self.audio = self.audio[self.audio != epsilon]

    def get_mel_spec(self, audio):
        return librosa.feature.melspectrogram(y=audio, sr=self.sr)

    def get_partial_audio(self, start_sec=None, end_sec=None):
        start_index = int(self.sr * start_sec) if start_sec is not None else 0
        end_index = int(self.sr * end_sec) if end_sec is not None else len(self.audio)
        return self.audio[start_index: end_index]

    def get_partial_mel_spec(self, start_sec, end_sec):
        audio = self.get_partial_audio(start_sec, end_sec)
        return self.get_mel_spec(audio)

    def get_chroma_cens(self, audio):
        return librosa.feature.chroma_cens(y=audio, sr=self.sr)

    def get_chroma_stft(self, audio):
        return librosa.feature.chroma_stft(y=audio, sr=self.sr)

    def get_partial_chroma_stft(self, start_sec=None, end_sec=None):
        audio = self.get_partial_audio(start_sec, end_sec)
        return self.get_chroma_stft(audio)

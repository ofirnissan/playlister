import os
import librosa


class Song:
    def __init__(self, filepath, epsilon=0, remove_zero_amp=True):
        self.song_name = filepath.split(os.sep)[-1].split('.')
        self.sr = None
        self.audio = None
        self.audio_path = filepath
        self._load_audio(filepath)

        if remove_zero_amp:
            self.remove_zero_amplitude(epsilon=epsilon)

    def _load_audio(self, filepath):
        audio, sr = librosa.load(filepath, sr=48000)
        self.audio = audio
        self.sr = sr

    def remove_zero_amplitude(self, epsilon=0):
        self.audio = self.audio[self.audio != epsilon]

    def get_mel_spec(self, audio, win_length_sec=0.025, hop_length_sec=0.01, n_filters=80):
        window_length_samples = int(win_length_sec * self.sr)
        hop_length_samples = int(hop_length_sec * self.sr)
        n_fft = window_length_samples

        return librosa.feature.melspectrogram(y=audio, sr=self.sr,
                                              win_length=window_length_samples,
                                              hop_length=hop_length_samples,
                                              n_fft=n_fft, n_mels=n_filters)

    def get_partial_audio(self, start_sec=None, end_sec=None):
        start_index = int(self.sr * start_sec) if start_sec is not None else 0
        end_index = int(self.sr * end_sec) if end_sec is not None else len(self.audio)
        return self.audio[start_index: end_index]

    def get_partial_mel_spec(self, start_sec, end_sec, win_length_sec=0.025, hop_length_sec=0.01, n_filters=80):
        audio = self.get_partial_audio(start_sec, end_sec)
        return self.get_mel_spec(audio, win_length_sec=win_length_sec, hop_length_sec=hop_length_sec,
                                 n_filters=n_filters)

    def get_chroma_cens(self, audio):
        return librosa.feature.chroma_cens(y=audio, sr=self.sr)

    def get_chroma_stft(self, audio):
        return librosa.feature.chroma_stft(y=audio, sr=self.sr)

    def get_partial_chroma_stft(self, start_sec=None, end_sec=None):
        audio = self.get_partial_audio(start_sec, end_sec)
        return self.get_chroma_stft(audio)


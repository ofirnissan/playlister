import os
import librosa
import numpy as np
from spleeter.separator import Separator
import soundfile as sf
import sys
sys.path.append("playlister/audio_hw2_code_only")
from plotter import Plotter


class Song:
    def __init__(self, filepath, epsilon=0, remove_zero_amp=True, sr=48000, win_length_sec=0.02, hop_length_sec=0.01):
        self.song_name = filepath.split(os.sep)[-1].split('.')[0]
        self.file_type = filepath.split(os.sep)[-1].split('.')[1]

        self.sr = None
        self.audio = None

        self.suffix_vocals = None # type: Song
        self.prefix_vocals = None # type: Song
        self.suffix_accompaniment = None # type: Song
        self.prefix_accompaniment = None # type: Song

        self.win_length_sec = win_length_sec
        self.hop_length_sec = hop_length_sec
        self.suffix_vocals_energy = None
        self.prefix_vocals_energy = None
        self.t_suffix = None
        self.t_prefix = None

        self.audio_path = filepath
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


    def find_vocals_and_accompaniment_for_suffix_and_prefix(self, time_in_sec=30):
        self.find_vocals_and_accompaniment(time_in_sec=time_in_sec, suffix=True)
        self.find_vocals_and_accompaniment(time_in_sec=time_in_sec, suffix=False)


    def find_vocals_and_accompaniment(self, time_in_sec=30, suffix=True):
        separator = Separator('spleeter:2stems')
        suffix_or_prefix = "suffix" if suffix else "prefix"

        # Save the partial audio to a temporary file
        path = f'{suffix_or_prefix}_{time_in_sec}_sec.wav'
        tmp_partial_audio = self.get_partial_audio(start_sec=-time_in_sec) if suffix else self.get_partial_audio(end_sec=time_in_sec)
        sf.write(path, tmp_partial_audio, self.sr)

        # Split the audio to vocals and accompaniment
        output_split_dir = f"spleeter_output/{self.song_name}"
        separator.separate_to_file(path, output_split_dir)

        # Remove the temporary audio file
        os.remove(path)
        dir_name = path.split('.')[0]
        # Load the vocals and accompaniment to Song objects
        home_dir_path = "/mnt/c/Users/ofirn/Documents/oni/elec/playlister"
        dir_path = os.path.join(home_dir_path, output_split_dir)
        dir_path = os.path.join(dir_path, dir_name)
        vocals_path = os.path.join(dir_path, 'vocals.wav')
        print("VOCALS_PATH")
        print(vocals_path)
        accompaniment_path = os.path.join(dir_path, 'accompaniment.wav')
        print(accompaniment_path)
        vocals = Song(vocals_path, remove_zero_amp=False)
        accompaniment = Song(accompaniment_path, remove_zero_amp=False)

        if suffix:
            self.suffix_vocals = vocals
            self.suffix_accompaniment = accompaniment
        else:
            self.prefix_vocals = vocals
            self.prefix_accompaniment = accompaniment


    def get_prefix_and_suffix_energy_array(self):
        assert self.suffix_vocals is not None and \
        self.prefix_vocals is not None, \
        "Please call find_vocals_and_accompaniment_for_suffix_and_prefix() before calling this function"
        self.suffix_vocals_energy, self.t_suffix = self.get_audio_energy_array(suffix=True)
        self.prefix_vocals_energy, self.t_prefix = self.get_audio_energy_array()
        

    def get_audio_energy_array(self, suffix=False):
        window_length_samples = int(self.win_length_sec * self.sr)
        hop_length_samples = int(self.hop_length_sec * self.sr)
        audio = self.suffix_vocals.audio if suffix else self.prefix_vocals.audio
        energy = np.array([
            sum(abs(audio[i:i + window_length_samples] ** 2))
            for i in range(0, len(audio) - window_length_samples, hop_length_samples)
        ])
        energy = 10 * np.log10(energy)
        t = librosa.frames_to_time(range(len(energy)), sr=self.sr, hop_length=hop_length_samples)
        return energy, t



import os
import librosa
import numpy as np
import soundfile as sf
from plotter import Plotter


class Song:
    def __init__(self, filepath, seperator=None, epsilon=0, remove_zero_amp=True, sr=48000, win_length_sec=0.02, hop_length_sec=0.01):
        self.song_name = filepath.split(os.sep)[-1].split('.')[0]
        self.file_type = filepath.split(os.sep)[-1].split('.')[1]
    
        self.sr = None
        self.audio = None

        self.partial_audio_time_in_sec = 30
        self.seperator = seperator
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
        if remove_zero_amp:
            self.remove_zero_amplitude(epsilon=epsilon)
        self.plotter = Plotter(self.audio, self.sr)
        self.audio_length = len(self.audio) / sr
        self.tempo = None

    def _load_audio(self, filepath, sr=48000):
        audio, sr = librosa.load(filepath, sr=sr)
        self.audio = audio
        self.sr = sr

    def remove_zero_amplitude(self, epsilon=0):
        self.audio = self.audio[self.audio != epsilon]
    
    def calc_tempo(self):
        self.tempo = librosa.beat.beat_track(y=self.audio, sr=16000)[0]

    def get_mel_spec(self, audio):
        return librosa.feature.melspectrogram(y=audio, sr=self.sr)

    def get_partial_audio(self, start_sec=None, end_sec=None, audio=None):
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

    def get_partial_several_intervals(self, intervals, product='chroma_stft', func='concat'):
        if product == 'chroma_stft' and func == 'concat':
            chroma_stft = self.get_partial_chroma_stft(intervals[0][0], intervals[0][1])
            if len(intervals) == 1:
                return chroma_stft
            for interval in intervals[1:]:
                chroma_stft = np.concatenate((chroma_stft, self.get_partial_chroma_stft(interval[0], interval[1])), axis=1)
            return chroma_stft

    def find_vocals_and_accompaniment_for_suffix_and_prefix(self, dir_path=None):
        # if dir path is not None search for the vocals and accompaniment in the given directory of both suffix and prefix
        if dir_path is not None:
            vocals_suffix_path = os.path.join(dir_path, f'suffix_{self.partial_audio_time_in_sec}_sec/vocals.wav')
            accompaniment_suffix_path = os.path.join(dir_path, f'suffix_{self.partial_audio_time_in_sec}_sec/accompaniment.wav')
            vocals_prefix_path = os.path.join(dir_path, f'prefix_{self.partial_audio_time_in_sec}_sec/vocals.wav')
            accompaniment_prefix_path = os.path.join(dir_path, f'prefix_{self.partial_audio_time_in_sec}_sec/accompaniment.wav')
            # verify files exist
            if os.path.exists(vocals_suffix_path) and os.path.exists(accompaniment_suffix_path) and \
            os.path.exists(vocals_prefix_path) and os.path.exists(accompaniment_prefix_path):
                # if files exist load them to Song objects
                self.suffix_vocals = Song(vocals_suffix_path, sr=self.sr, remove_zero_amp=False)
                self.suffix_accompaniment = Song(accompaniment_suffix_path, sr=self.sr, remove_zero_amp=False)
                self.prefix_vocals = Song(vocals_prefix_path, sr=self.sr, remove_zero_amp=False)
                self.prefix_accompaniment = Song(accompaniment_prefix_path, sr=self.sr, remove_zero_amp=False)
                return
        # if dir path is None or the files do not exist in the given directory
        assert self.seperator is not None, "Please provide a seperator object"
        self.find_vocals_and_accompaniment(suffix=True, dir_path=dir_path)
        self.find_vocals_and_accompaniment(suffix=False, dir_path=dir_path)

    def find_vocals_and_accompaniment(self, suffix=True, dir_path=''):
        
        time_in_sec = self.partial_audio_time_in_sec
        separator = self.seperator
        suffix_or_prefix = "suffix" if suffix else "prefix"

        # Save the partial audio to a temporary file
        path = f'{dir_path}/{suffix_or_prefix}_{time_in_sec}_sec.wav'
        tmp_partial_audio = self.get_partial_audio(start_sec=-time_in_sec) if suffix else self.get_partial_audio(end_sec=time_in_sec)
        sf.write(path, tmp_partial_audio, self.sr)

        # Split the audio to vocals and accompaniment
        output_split_dir = f"{dir_path}"
        separator.separate_to_file(path, output_split_dir)

        # Remove the temporary audio file
        os.remove(path)

        # Load the vocals and accompaniment to Song object
        output_split_dir = os.path.join(output_split_dir, f'{suffix_or_prefix}_{time_in_sec}_sec')
        vocals_path = os.path.join(output_split_dir, 'vocals.wav')
        print("VOCAL_PATH")
        print(vocals_path)
        accompaniment_path = os.path.join(output_split_dir, 'accompaniment.wav')
        print("ACCOMPANIMENT_PATH")
        print(accompaniment_path)
        vocals = Song(vocals_path, sr=self.sr, remove_zero_amp=False)
        accompaniment = Song(accompaniment_path, sr=self.sr, remove_zero_amp=False)

        if suffix:
            self.suffix_vocals = vocals
            self.suffix_accompaniment = accompaniment
        else:
            self.prefix_vocals = vocals
            self.prefix_accompaniment = accompaniment

    def get_prefix_and_suffix_energy_array_post_separation(self):
        assert self.suffix_vocals is not None and \
        self.prefix_vocals is not None, \
        "Please call find_vocals_and_accompaniment_for_suffix_and_prefix() before calling this function"
        self.suffix_vocals_energy, self.t_suffix = self.get_audio_energy_array(self.suffix_vocals.audio)
        self.prefix_vocals_energy, self.t_prefix = self.get_audio_energy_array(self.prefix_vocals.audio)

    def get_audio_energy_array(self, audio):
        window_length_samples = int(self.win_length_sec * self.sr)
        hop_length_samples = int(self.hop_length_sec * self.sr)
        energy = np.array([
            sum(abs(audio[i:i + window_length_samples] ** 2))
            for i in range(0, len(audio) - window_length_samples, hop_length_samples)
        ])
        energy = 10 * np.log10(energy)
        t = librosa.frames_to_time(range(len(energy)), sr=self.sr, hop_length=hop_length_samples)
        return energy, t

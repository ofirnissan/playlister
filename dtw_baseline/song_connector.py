from song_handler import Song
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
sys.path.append("playlister/audio_hw2_code_only")
from plotter import Plotter
import math


home_dir_path = "/mnt/c/Users/ofirn/Documents/oni/elec/playlister"
noise_threshold = 0
min_size_of_time_interval = 2 # sec


def find_silent_intervals_of_partial(song: Song, threshold=0, min_time_interval_length=2, suffix=False):
    indices = np.where(song.suffix_vocals_energy < threshold)[0] if suffix else np.where(song.prefix_vocals_energy < threshold)[0]
    scilent_time_intervals = []
    start = song.t_suffix[indices[0]] if suffix else song.t_prefix[indices[0]]
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > 1 or i == len(indices) - 2:
            cur = song.t_suffix[indices[i]] if suffix else song.t_prefix[indices[i]]
            if cur - start > min_time_interval_length:
                scilent_time_intervals.append((start, cur))
            start = song.t_suffix[indices[i + 1]] if suffix else song.t_suffix[indices[i + 1]]
    return scilent_time_intervals


def find_silent_intervals_of_suffix_and_prefix(song: Song, threshold=0, min_time_interval_length=2):
    suffix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=True)
    prefix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=False)
    return prefix_scilent_intervals, suffix_scilent_intervals


def concat_between_songs_post_spleeter(song1: Song, song2: Song):
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    # Get scilent time intervals of prefix and suffix of both songs:
    song1_prefix_scilent_intervals, song1_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song1, threshold=noise_threshold, min_time_interval_length=min_size_of_time_interval)
    song2_prefix_scilent_intervals, song2_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song2, threshold=noise_threshold, min_time_interval_length=min_size_of_time_interval)
    # Get accompaniment of prefix and suffix of both songs:
    chroma_suffix_song1 = song1.suffix_accompaniment.get_partial_chroma_stft(start_sec=song1_suffix_scilent_intervals[0][0],\
                                                         end_sec=song1_suffix_scilent_intervals[0][1])
    chroma_prefix_song2 = song2.prefix_accompaniment.get_partial_chroma_stft(start_sec=song2_prefix_scilent_intervals[0][0],\
                                                                             end_sec=song2_prefix_scilent_intervals[0][1])

    dtw_cost_matrix, wp = librosa.sequence.dtw(chroma_suffix_song1, chroma_prefix_song2, subseq=True)  # wp is the Warping path
    # Get the index of the minimum cost in the path
    frames_in_sec = int(1 * song1.sr / 512)
    path_cost = dtw_cost_matrix[wp[:, 0], wp[:, 1]]
    idx = np.argmin(path_cost[:len(path_cost)-frames_in_sec] - path_cost[frames_in_sec:])    
    
    # Cut the audio in the right place and concat the two songs
    suffix_cut_frame, prefix_cut_frame = wp[idx]
    prefix_cut_audio_index = prefix_cut_frame * 512
    suffix_cut_audio_index = suffix_cut_frame * 512
    first = song1.get_partial_audio(end_sec=len(song1.audio)/song1.sr - song1.partial_audio_time_in_sec + song1_suffix_scilent_intervals[0][0] + suffix_cut_audio_index/song1.sr)
    second = song2.get_partial_audio(prefix_cut_audio_index/song2.sr + song2_prefix_scilent_intervals[0][0])
    new_audio = fadeout_cur_fadein_next(first, second, song1.sr, duration=3.0)

    # Save the new audio
    sf.write('concat_song.wav', new_audio, song1.sr)


def fadeout_cur_fadein_next(audio1, audio2, sr, duration=3.0):
    duration = 3.0
    apply_fadeout(audio1, sr, duration)
    apply_fadein(audio2, sr, duration)
    length = int(duration*sr)
    new_audio = audio1
    end = new_audio.shape[0]
    start = end - length
    new_audio[start:end] += audio2[:length]
    new_audio = np.concatenate((new_audio, audio2[length:]))
    return new_audio


def apply_fadeout(audio, sr, duration=3.0):
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def apply_fadein(audio, sr, duration=3.0):
    length = int(duration*sr)
    # linear fade
    fade_curve = np.linspace(0.0, 1.0, length)
    # apply the curve
    audio[:length] = audio[:length] * fade_curve

if __name__ == '__main__':
    song_names = ["Incubus - Drive", "bahaim hakol over"]
    songs = []
    for song_name in song_names:
        song = Song(f"/mnt/c/Users/ofirn/Music/songs/{song_name}.mp3")
        
        song.partial_audio_time_in_sec = 30
        time_in_sec = song.partial_audio_time_in_sec
        # song.find_vocals_and_accompaniment_for_suffix_and_prefix()
        
        suffix_song_vocals = Song(f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}/suffix_{time_in_sec}_sec/vocals.wav", remove_zero_amp=False)
        prefix_song_vocals = Song(f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}/prefix_{time_in_sec}_sec/vocals.wav", remove_zero_amp=False)
        song.prefix_vocals = prefix_song_vocals
        song.suffix_vocals = suffix_song_vocals

        suffix_song_accompaniment = Song(f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}/suffix_{time_in_sec}_sec/accompaniment.wav", remove_zero_amp=False)
        prefix_song_accompaniment = Song(f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}/prefix_{time_in_sec}_sec/accompaniment.wav", remove_zero_amp=False)
        song.prefix_accompaniment = prefix_song_accompaniment
        song.suffix_accompaniment = suffix_song_accompaniment

        song.get_prefix_and_suffix_energy_array()

        songs.append(song)
        # suffix_song.plotter.plot_energy_and_rms(threshold=0, plot_rms=True, title="Q3.a.i")
        # suffix_song.plotter.plot_mel_spectogram()
    
    concat_between_songs_post_spleeter(songs[0], songs[1])
    
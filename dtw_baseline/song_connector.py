from song_handler import Song
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


home_dir_path = "/mnt/c/Users/ofirn/Documents/oni/elec/playlister"

def find_min_pair_of_cost_matrix_trace(dtw_cost_matrix, wp):
    trace_data = dtw_cost_matrix[wp[:, 0], wp[:, 1]]
    return wp[trace_data.argmin()]


def connect_between_songs_chroma_feature(song1: Song, song2: Song, show_cost_matrix_and_wp=False):
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    time_in_sec = 60
    suffix = song1.get_partial_chroma_stft(start_sec=-time_in_sec)  # np.ndarray
    prefix = song2.get_partial_chroma_stft(end_sec=time_in_sec)  # np.ndarray
    dtw_cost_matrix, wp = librosa.sequence.dtw(suffix, prefix, subseq=True)  # wp is the Warping path
    # idx = np.random.choice(len(wp), 1)[0]
    idx = len(wp) - 1
    suffix_cut_frame, prefix_cut_frame = wp[idx]
    prefix_cut_audio_index = prefix_cut_frame * 512
    suffix_cut_audio_index = len(song1.audio) - (len(song1.get_partial_audio(start_sec=-time_in_sec)) - suffix_cut_frame * 512)

    new_audio = np.concatenate((song1.audio[: suffix_cut_audio_index], song2.audio[prefix_cut_audio_index:]))
    sf.write('concat_song.wav', new_audio, song1.sr)

    if show_cost_matrix_and_wp:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(dtw_cost_matrix, x_axis='frames', y_axis='frames', ax=ax, hop_length=512)
        ax.set(title='DTW cost', xlabel='prefix', ylabel='sufix')
        ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
        ax.legend()
        fig.colorbar(img, ax=ax)
        plt.show()


if __name__ == '__main__':
    # song1 = Song("/mnt/c/Users/ofirn/Music/songs/Green Eyes.mp3")
    # song2 = Song("/mnt/c/Users/ofirn/Music/songs/In My Place.mp3")
    song3 = Song("/mnt/c/Users/ofirn/Music/songs/Red Hot Chili Peppers - The Adventures of Rain Dance Maggie.mp3")
    # song2.find_vocals_and_accompaniment_for_suffix_and_prefix()
    # song1.find_vocals_and_accompaniment_for_suffix_and_prefix()
    song3.find_vocals_and_accompaniment_for_suffix_and_prefix()
    


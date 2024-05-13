from song_handler import Song
import soundfile as sf
import librosa
import numpy as np
import sys
sys.path.append("playlister/audio_hw2_code_only")
from plotter import Plotter
from spleeter.separator import Separator


home_dir_path = "/mnt/c/Users/ofirn/Documents/oni/elec/playlister"
noise_threshold = 0 # dB ; Threshold for silence detection in the vocals energy array.
min_size_of_time_interval = 2 # sec ; Minimum size of time interval for silence detection.
jump = 2 # Minimum number of frames between two silent intervals to consider them as different intervals. 
          # This is used to deal with noise in the energy array of the vocals.
fade_duration = 3.0 # sec ; Duration of the fade in and fade out effects.



def connect_between_songs_by_dtw_only(song1: Song, song2: Song):
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    suffix_audio = song1.get_partial_audio(start_sec=-song1.partial_audio_time_in_sec)
    prefix_audio = song2.get_partial_audio(end_sec=song2.partial_audio_time_in_sec)

    # suffix_audio = song1.suffix_accompaniment.audio
    # prefix_audio = song2.prefix_accompaniment.audio

    suffix_mel_spec = song1.get_chroma_stft(suffix_audio)  # np.ndarray
    prefix_mel_spec = song2.get_chroma_stft(prefix_audio)  # np.ndarray

    # suffix_xs, suffix_pitch = song1.get_pitch(suffix_audio)
    # prefix_xs, prefix_pitch = song2.get_pitch(prefix_audio)
    
    # p = Plotter(song1.get_partial_audio(start_sec=-time_in_sec), sr=song1.sr)
    # p.plot_spectogram()

    dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_mel_spec, prefix_mel_spec, subseq=True)  # wp is the Warping path
    # idx = np.random.choice(len(wp), 1)[0]
    frames_in_sec = int(5 * song1.sr / 512)
    path_cost = dtw_cost_matrix[wp[:, 0], wp[:, 1]]
    min_index1 = np.argmin(path_cost[:len(path_cost) - frames_in_sec] - path_cost[frames_in_sec:])
    # propagation = np.cumsum(np.sum(wp[:-1] - wp[1:], axis=1))
    # min_index2 = np.argmax(propagation[frames_in_sec:] - propagation[:len(propagation) - frames_in_sec]) + frames_in_sec + 1
    idx = min_index1
    suffix_cut_frame, prefix_cut_frame = wp[idx]
    prefix_cut_audio_index = prefix_cut_frame * 512
    suffix_cut_audio_index = len(song1.audio) - (len(suffix_audio) - suffix_cut_frame * 512)

    new_audio = fadeout_cur_fadein_next(song1.audio[: suffix_cut_audio_index], song2.audio[prefix_cut_audio_index:], song1.sr, duration=fade_duration)

    sf.write('concat_song.wav', new_audio, song1.sr)



def find_silent_intervals_of_partial(song: Song, threshold=0, min_time_interval_length=2, suffix=False):
    indices = np.where(song.suffix_vocals_energy < threshold)[0] if suffix else np.where(song.prefix_vocals_energy < threshold)[0]
    scilent_time_intervals = []
    start = song.t_suffix[indices[0]] if suffix else song.t_prefix[indices[0]]
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > jump or i == len(indices) - 2:
            cur = song.t_suffix[indices[i]] if suffix else song.t_prefix[indices[i]]
            if cur - start > min_time_interval_length:
                scilent_time_intervals.append((start, cur))
            start = song.t_suffix[indices[i + 1]] if suffix else song.t_suffix[indices[i + 1]]
    return scilent_time_intervals


def find_silent_intervals_of_suffix_and_prefix(song: Song, threshold=0, min_time_interval_length=2):
    suffix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=True)
    if len(suffix_scilent_intervals) == 0:
        suffix_scilent_intervals = [(song.partial_audio_time_in_sec-min_size_of_time_interval, song.partial_audio_time_in_sec)]
    prefix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=False)
    if len(prefix_scilent_intervals) == 0:
        prefix_scilent_intervals = [(0, min_size_of_time_interval)]
    return prefix_scilent_intervals, suffix_scilent_intervals


def show_dtw_cost_matrix_and_wp(dtw_cost_matrix, wp):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    img = librosa.display.specshow(dtw_cost_matrix, x_axis='frames', y_axis='frames', ax=ax, hop_length=512)
    ax.set(title='DTW cost', xlabel='prefix', ylabel='sufix')
    ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax.legend()
    fig.colorbar(img, ax=ax)
    plt.show()


def concat_between_songs_post_spleeter(song1: Song, song2: Song, song_builder: Song = None):
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    # Get scilent time intervals of prefix and suffix of both songs:
    song1_prefix_scilent_intervals, song1_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song1, threshold=noise_threshold, min_time_interval_length=min_size_of_time_interval)
    song2_prefix_scilent_intervals, song2_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song2, threshold=noise_threshold, min_time_interval_length=min_size_of_time_interval)
    # Get accompaniment of prefix and suffix of both songs:
    chroma_suffix_song1 = song1.suffix_accompaniment.get_partial_chroma_stft(start_sec=song1_suffix_scilent_intervals[0][0],\
                                                         end_sec=song1_suffix_scilent_intervals[0][1])
    chroma_prefix_song2 = song2.prefix_accompaniment.get_partial_chroma_stft(start_sec=song2_prefix_scilent_intervals[-1][0],\
                                                                             end_sec=song2_prefix_scilent_intervals[-1][1])
    # chroma_suffix_song1 = song1.suffix_accompaniment.get_partial_several_intervals(song1_suffix_scilent_intervals)
    # chroma_prefix_song2 = song2.prefix_accompaniment.get_partial_several_intervals(song2_prefix_scilent_intervals)
    
    dtw_cost_matrix, wp = librosa.sequence.dtw(chroma_suffix_song1, chroma_prefix_song2, subseq=True)  # wp is the Warping path
    # Get the index of the minimum cost in the path
    frames_in_sec = int(1 * song1.sr / 512)
    wp_first_col = wp[:, 0] if max(wp[:, 0]) == dtw_cost_matrix.shape[0] - 1 else wp[:, 1]
    wp_second_col = wp[:, 1] if max(wp[:, 0]) == dtw_cost_matrix.shape[0] - 1 else wp[:, 0]
    path_cost = dtw_cost_matrix[wp_first_col, wp_second_col]
    idx = np.argmin(path_cost[:len(path_cost)-frames_in_sec] - path_cost[frames_in_sec:])    
      
    # Cut the audio in the right place and concat the two songs
    suffix_cut_frame, prefix_cut_frame = wp[idx]
    prefix_cut_audio_index = prefix_cut_frame * 512
    suffix_cut_audio_index = suffix_cut_frame * 512
    first_song = song_builder if song_builder is not None else song1
    first = first_song.get_partial_audio(end_sec=len(first_song.audio)/song1.sr - song1.partial_audio_time_in_sec + song1_suffix_scilent_intervals[0][0] + suffix_cut_audio_index/song1.sr)
    second = song2.get_partial_audio(prefix_cut_audio_index/song2.sr + song2_prefix_scilent_intervals[-1][0])
    new_audio = fadeout_cur_fadein_next(first, second, song1.sr, duration=fade_duration)


    sf.write('concat_song.wav', new_audio, first_song.sr)
    # TODO: create a new song object with the new audio without saving it to a file
    new_song = Song('/mnt/c/Users/ofirn/Documents/oni/elec/playlister/concat_song.wav')

    return new_song

    # Save the new audio
    # sf.write('concat_song.wav', new_audio, song1.sr)


def organize_song_list_using_tempo(songs):
    # Sort the songs by tempo
    songs = sorted(songs, key=lambda x: x.tempo)
    print([song.song_name for song in songs])
    return songs


def fadeout_cur_fadein_next(audio1, audio2, sr, duration=fade_duration):
    apply_fadeout(audio1, sr, duration)
    apply_fadein(audio2, sr, duration)
    length = int(duration*sr)
    new_audio = audio1
    end = new_audio.shape[0]
    start = end - length
    new_audio[start:end] += audio2[:length]
    new_audio = np.concatenate((new_audio, audio2[length:]))
    return new_audio


def apply_fadeout(audio, sr, duration=fade_duration):
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def apply_fadein(audio, sr, duration=fade_duration):
    length = int(duration*sr)
    # linear fade
    fade_curve = np.linspace(0.0, 1.0, length)
    # apply the curve
    audio[:length] = audio[:length] * fade_curve

if __name__ == '__main__':
    # , 'Incubus - Drive', 'biladaih'
    song_names = ['Incubus - Drive', 'Wish You Were Here - Incubus - Lyrics']
    # song_names = ['lev_shel_gever', 'biladaih']
    songs = []
    sep = Separator('spleeter:2stems')
    # sep = None
    for song_name in song_names:
        print(f"parsing: {song_name}")

        song = Song(f"/mnt/c/Users/ofirn/Music/songs/{song_name}.mp3", seperator=sep, remove_zero_amp=True)
        song.partial_audio_time_in_sec = 60
        time_in_sec = song.partial_audio_time_in_sec
        # song.calc_tempo()
        song.find_vocals_and_accompaniment_for_suffix_and_prefix(dir_path=f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}")
        song.get_prefix_and_suffix_energy_array_post_separation()
        songs.append(song)
        
        # suffix_song_vocals.plotter.plot_energy_and_rms(threshold=noise_threshold, plot_rms=True, title="suffix_vocal")
        # prefix_song_vocals.plotter.plot_energy_and_rms(threshold=noise_threshold, plot_rms=True, title="prefix_vocal")

        # suffix_song.plotter.plot_mel_spectogram()

    # connect_between_songs_by_dtw_only(songs[0], songs[1])

    # songs = organize_song_list_using_tempo(songs)
    new_song = songs[0]
    for i in range(len(songs)-1):
        new_song = concat_between_songs_post_spleeter(songs[i], songs[i+1], song_builder=new_song)
    sf.write('concat_song.wav', new_song.audio, new_song.sr)
    
    
from song_handler import Song
import soundfile as sf
import librosa
import numpy as np
import os
from utils import Graph, fadeout_cur_fadein_next, get_partial_audio
from utils import show_dtw_cost_matrix_and_wp  # for debug


HOME_PATH = '/home/joberant/NLP_2324/yaelshemesh'
DEFAULT_OUT_PATH = "outputs"

SONGS_LIST_DIR = 'yael_playlist'
NO_VOCAL_DETECTION_THRESHOLD = 5  # dB ; Threshold for silence detection in the vocals energy array.
NOISE_THRESHOLD_FOR_TRANSFORMATION = 0  # dB ; Threshold for silence detection in transformation.
MIN_SIZE_OF_TIME_INTERVAL = 2  # sec ; Minimum size of time interval for silence detection.
JUMP = 2  # Minimum number of frames between two silent intervals to consider them as different intervals.
# This is used to deal with noise in the energy array of the vocals.
FADE_DURATION = 2.0  # sec ; Duration of the fade in and fade out effects.


def connect_between_songs_by_dtw_only(song1: Song, song2: Song, file_path: str, song_builder: Song=None,
                                      accompaniment=False, pick_idx_method='min_dist_in_path',
                                      fade_duration=FADE_DURATION):
    print(f"Concat {song1.song_name} and {song2.song_name} using DTW method (over chroma stft of the songs)")
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    
    # Get suffix of song1 and prefix of song2
    suffix_audio = song1.get_partial_audio(start_sec=-song1.partial_audio_time_in_sec)
    prefix_audio = song2.get_partial_audio(end_sec=song2.partial_audio_time_in_sec)
    if accompaniment: # Use accompaniment audio only (if it has been separated)
        suffix_audio = song1.suffix_accompaniment.audio
        prefix_audio = song2.prefix_accompaniment.audio

    # Get the chroma features of the audio
    suffix_chroma_stft = song1.get_chroma_stft(suffix_audio)  # np.ndarray
    prefix_chroma_stft = song2.get_chroma_stft(prefix_audio)  # np.ndarray

    # Use DTW to find the best match between suffix and prefix:
    sorted_indexes = []
    window_in_sec = int(5 * song1.sr / 512) 
    dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_chroma_stft, prefix_chroma_stft, subseq=True)  # wp is the Warping path
    if pick_idx_method == 'random':
        sorted_indexes = np.random.choice(len(wp), len(wp))
    elif pick_idx_method == 'max_propagation':
        propagation = np.cumsum(np.sum(wp[:-1] - wp[1:], axis=1))
        sorted_indexes = np.argsort(propagation[window_in_sec:] - propagation[:len(propagation) - window_in_sec])[::-1]
    elif pick_idx_method == 'min_dist_in_path':
        path_cost = dtw_cost_matrix[wp[:, 0], wp[:, 1]]
        sorted_indexes = np.argsort(path_cost[:len(path_cost) - window_in_sec] - path_cost[window_in_sec:])
    
    # Among chosen indeces by the above methods, choose index in which suffix is not scilent:
    first_audio = None
    second_audio = None
    first_song = song_builder if song_builder is not None else song1
    best_tuple_in_sec = None

    for idx in sorted_indexes:
        idx += window_in_sec  # wp start from max to min. By increasing idx by window, we get the beginning of the match
        suffix_cut_frame, prefix_cut_frame = wp[idx]
        prefix_cut_audio_index = prefix_cut_frame * 512
        suffix_cut_audio_index = len(first_song.audio) - (len(suffix_audio) - suffix_cut_frame * 512)
        first_audio = first_song.audio[: suffix_cut_audio_index]
        second_audio = song2.audio[prefix_cut_audio_index:]
        if first_song.get_audio_energy_array(get_partial_audio(first_audio, first_song.sr, start_sec=-2))[0].mean()\
                > NOISE_THRESHOLD_FOR_TRANSFORMATION:
            best_tuple_in_sec = (suffix_cut_audio_index / first_song.sr, prefix_cut_audio_index / song2.sr)
            break

    # Concat songs with overlapping fade and save new song:
    new_audio = fadeout_cur_fadein_next(first_audio, second_audio, song1.sr, duration=fade_duration)
    # new_audio = np.concatenate([first_audio, second_audio])
    sf.write(file_path, new_audio, first_song.sr)
    new_song = Song(file_path)
    return new_song, best_tuple_in_sec


def find_silent_intervals_of_partial(song: Song, threshold=0, min_time_interval_length=2, suffix=False):
    indices = np.where(song.suffix_vocals_energy < threshold)[0] if suffix else np.where(song.prefix_vocals_energy < threshold)[0]
    scilent_time_intervals = []
    start = song.t_suffix[indices[0]] if suffix else song.t_prefix[indices[0]]
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > JUMP or i == len(indices) - 2:
            cur = song.t_suffix[indices[i]] if suffix else song.t_prefix[indices[i]]
            if cur - start > min_time_interval_length:
                scilent_time_intervals.append((start, cur))
            start = song.t_suffix[indices[i + 1]] if suffix else song.t_suffix[indices[i + 1]]
    return scilent_time_intervals


def find_silent_intervals_of_suffix_and_prefix(song: Song, threshold=0, min_time_interval_length=2):
    suffix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=True)
    if len(suffix_scilent_intervals) == 0:
        suffix_scilent_intervals = [(song.partial_audio_time_in_sec - MIN_SIZE_OF_TIME_INTERVAL, song.partial_audio_time_in_sec)]
    prefix_scilent_intervals = find_silent_intervals_of_partial(song, threshold, min_time_interval_length, suffix=False)
    if len(prefix_scilent_intervals) == 0:
        prefix_scilent_intervals = [(0, MIN_SIZE_OF_TIME_INTERVAL)]
    return prefix_scilent_intervals, suffix_scilent_intervals


def dtw_over_songs_intervals_post_spleeter(song1: Song, song2: Song, song1_suffix_scilent_intervals, song2_prefix_scilent_intervals):
    min_value = np.inf
    window_in_sec = int(1 * song1.sr / 512) # we take window of 1 sec 
    min_idx = 0
    min_i = 0
    min_j = -1
    for i in range(len(song1_suffix_scilent_intervals)):
        for j in range(len(song2_prefix_scilent_intervals)):
            chroma_suffix_song1 = song1.suffix_accompaniment.get_partial_chroma_stft(start_sec=song1_suffix_scilent_intervals[i][0],\
                                                                end_sec=song1_suffix_scilent_intervals[i][1])
            chroma_prefix_song2 = song2.prefix_accompaniment.get_partial_chroma_stft(start_sec=song2_prefix_scilent_intervals[j][0],\
                                                                                    end_sec=song2_prefix_scilent_intervals[j][1])
            dtw_cost_matrix, wp = librosa.sequence.dtw(chroma_suffix_song1, chroma_prefix_song2, subseq=True)  # wp is the Warping path
            # Get the index of the minimum cost in the path
            wp_first_col = wp[:, 0] if max(wp[:, 0]) == dtw_cost_matrix.shape[0] - 1 else wp[:, 1]
            wp_second_col = wp[:, 1] if max(wp[:, 0]) == dtw_cost_matrix.shape[0] - 1 else wp[:, 0]
            path_cost = dtw_cost_matrix[wp_first_col, wp_second_col]
            distances_in_path = path_cost[:len(path_cost)-window_in_sec] - path_cost[window_in_sec:]
            sorted_indeces = np.argsort(distances_in_path)
            # Make sure that the suffix is not scilent:
            first_audio = None
            for idx in sorted_indeces:
                suffix_cut_frame = wp[idx][0]
                suffix_cut_audio_index = suffix_cut_frame * 512
                first_audio = song1.get_partial_audio(end_sec=len(song1.audio)/song1.sr - song1.partial_audio_time_in_sec + song1_suffix_scilent_intervals[min_i][0] + suffix_cut_audio_index/song1.sr)
                if song1.get_audio_energy_array(get_partial_audio(first_audio, song1.sr, start_sec=-2))[0].mean() > NOISE_THRESHOLD_FOR_TRANSFORMATION:
                    break
            if distances_in_path[idx] < min_value:
                min_value = distances_in_path[idx]
                min_idx = idx
                min_i = i
                min_j = j
    return min_i, min_j, wp[min_idx]


def concat_between_songs_post_spleeter(song1: Song, song2: Song, file_path: str, song_builder: Song = None,
                                       fade_duration=FADE_DURATION):
    print(f"Concat {song1.song_name} and {song2.song_name} using vocals and accompaniment separation then dtw")
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    
    # Get scilent time intervals of prefix and suffix of both songs:
    song1_prefix_scilent_intervals, song1_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song1, threshold=NO_VOCAL_DETECTION_THRESHOLD, min_time_interval_length=MIN_SIZE_OF_TIME_INTERVAL)
    song2_prefix_scilent_intervals, song2_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song2, threshold=NO_VOCAL_DETECTION_THRESHOLD, min_time_interval_length=MIN_SIZE_OF_TIME_INTERVAL)
    
    print(f"{song1.song_name} suffix_scilent_intervals: ", song1_suffix_scilent_intervals)
    print(f"{song2.song_name} prefix_scilent_intervals: ", song2_prefix_scilent_intervals)
    
    # Find the best interval to cut the audio in the right place and concat the two songs
    min_i, min_j, wp_min_idx = \
        dtw_over_songs_intervals_post_spleeter\
            (song1, song2, song1_suffix_scilent_intervals, song2_prefix_scilent_intervals)
    
    # Cut the audio in the right place and concat the two songs
    suffix_cut_frame, prefix_cut_frame = wp_min_idx
    prefix_cut_audio_index = prefix_cut_frame * 512
    suffix_cut_audio_index = suffix_cut_frame * 512
    first_song = song_builder if song_builder is not None else song1
    first = first_song.get_partial_audio(end_sec=len(first_song.audio)/song1.sr - song1.partial_audio_time_in_sec + song1_suffix_scilent_intervals[min_i][0] + suffix_cut_audio_index/song1.sr)
    second = song2.get_partial_audio(prefix_cut_audio_index/song2.sr + song2_prefix_scilent_intervals[min_j][0])
    new_audio = fadeout_cur_fadein_next(first, second, song1.sr, duration=fade_duration)
    # new_audio = np.concatenate([first, second])

    sf.write(file_path, new_audio, first_song.sr)
    new_song = Song(file_path)
    return new_song


def organize_song_list_using_dtw_greedy(songs):
    print("organizing songs using dtw greedy")
    cur_song = songs[0] # assume first song is the first song in the playlist
    songs_set = set(songs)  # set of songs remain to be organized
    songs_set.remove(cur_song) 
    organized_songs = []  # list of songs in the organized playlist
    organized_songs.append(cur_song)
    
    while len(songs_set) > 0:
        best_dtw_cost = np.inf # best dtw cost between two songs
        best_song = None # best song to be added
        for next_song in songs_set:
            # calculate dtw of cur song suffix with all other songs prefix
            suffix_audio = cur_song.get_partial_audio(start_sec=-cur_song.partial_audio_time_in_sec)
            prefix_audio = next_song.get_partial_audio(end_sec=next_song.partial_audio_time_in_sec)
            suffix_chroma_stft = cur_song.get_chroma_stft(suffix_audio)  
            prefix_chroma_stft = next_song.get_chroma_stft(prefix_audio) 
            dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_chroma_stft, prefix_chroma_stft, subseq=True)
            dtw_cost = dtw_cost_matrix[-1,-1] 
            if dtw_cost < best_dtw_cost:
                best_dtw_cost = dtw_cost
                best_song = next_song
        # add the best song to the organized playlist and remove it from the set of songs to be organized
        organized_songs.append(best_song)
        songs_set.remove(best_song)
        cur_song = best_song
    return organized_songs


def organize_song_list_using_dtw_optimum_approximation(songs):
    # create graph of songs with edges between each two songs with the cost of the dtw between them
    n = len(songs)
    g = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            # calculate dtw of cur song suffix with all other songs prefix
            suffix_audio = songs[i].get_partial_audio(start_sec=-songs[i].partial_audio_time_in_sec)
            prefix_audio = songs[j].get_partial_audio(end_sec=songs[j].partial_audio_time_in_sec)
            suffix_chroma_stft = songs[i].get_chroma_stft(suffix_audio)  
            prefix_chroma_stft = songs[j].get_chroma_stft(prefix_audio) 
            dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_chroma_stft, prefix_chroma_stft, subseq=True)
            dtw_cost = dtw_cost_matrix[-1,-1] 
            g.graph[i][j] = dtw_cost
            # calculate dtw of cur song prefix with all other songs suffix
            suffix_audio = songs[j].get_partial_audio(start_sec=-songs[j].partial_audio_time_in_sec)
            prefix_audio = songs[i].get_partial_audio(end_sec=songs[i].partial_audio_time_in_sec)
            suffix_chroma_stft = songs[j].get_chroma_stft(suffix_audio)  
            prefix_chroma_stft = songs[i].get_chroma_stft(prefix_audio) 
            dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_chroma_stft, prefix_chroma_stft, subseq=True)
            dtw_cost = dtw_cost_matrix[-1,-1] 
            g.graph[j][i] = dtw_cost
    organized_songs_indexes = g.find_approximate_optimal_tsp_path()
    return [songs[idx] for idx in organized_songs_indexes]
    

def organize_song_list_using_tempo(songs):
    # Sort the songs by tempo
    songs = sorted(songs, key=lambda x: x.tempo)
    return songs


def create_full_playlist_using_dtw(outpath, songs_list_dir, home_directory, use_spleeter=False, save_numpy_arrays=False,
                                   fade_duration=FADE_DURATION):
    song_names = os.listdir(songs_list_dir)
    songs = []
    sep = None
    spleeter_output_dir_path = os.path.join(home_directory, 'spleeter_output')
    if use_spleeter:
        from spleeter.separator import Separator
        sep = Separator('spleeter:2stems')
        os.makedirs(spleeter_output_dir_path, exist_ok=True)

    assert len(song_names) != 0
    # Get all files from dir to song_names list:
    for song_name in song_names:
        print(f"parsing: {song_name}")
        song = Song(os.path.join(songs_list_dir, f"{song_name}"),  seperator=sep, remove_zero_amp=True)
        song.partial_audio_time_in_sec = 45
        # song.calc_tempo()
        if use_spleeter:
            os.makedirs(os.path.join(spleeter_output_dir_path, song_name),  exist_ok=True)
            song.find_vocals_and_accompaniment_for_suffix_and_prefix(dir_path=os.path.join(spleeter_output_dir_path,
                                                                                           song_name))
            song.get_prefix_and_suffix_energy_array_post_separation()
        songs.append(song)

    # Set songs order and print result:
    songs = organize_song_list_using_dtw_optimum_approximation(songs)
    # songs = organize_song_list_using_tempo(songs)
    # songs = organize_song_list_using_dtw_greedy(songs)
    print([song.song_name for song in songs])
    new_song = songs[0]
    print('concatinating songs...')
    result_fp = os.path.join(outpath, 'dtw_concat_playlist.wav')
    tuples_cuts_in_sec = []
    for i in range(len(songs) - 1):
        if use_spleeter:
            new_song = concat_between_songs_post_spleeter(songs[i], songs[i+1], file_path=result_fp,
                                                          song_builder=new_song, fade_duration=fade_duration)
            # new_song = connect_between_songs_by_dtw_only(songs[i], songs[i+1], file_path=result_fp,
            #                                              song_builder=new_song, accompaniment=True)

        else:
            new_song, best_tuple_cuts_in_sec = connect_between_songs_by_dtw_only(songs[i], songs[i + 1],
                                                                                 file_path=result_fp,
                                                                                 song_builder=new_song,
                                                                                 fade_duration=fade_duration)
            tuples_cuts_in_sec.append(best_tuple_cuts_in_sec)
    sf.write(result_fp, new_song.audio, new_song.sr)
    if save_numpy_arrays:
        np.save(os.path.join(outpath, f'baseline_playlister_playlist_numpy.npy'), new_song.audio)
        np.save(os.path.join(outpath, f'baseline_songs_name_order.npy'), np.array([song.song_name for song in songs]))
        np.save(os.path.join(outpath, f'best_cuts_in_sec.npy'), np.array(tuples_cuts_in_sec))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Playlister DTW baseline implementation. Saves the playlist as one audio file.')
    parser.add_argument("--home_dir", type=str, default=HOME_PATH, help='split songs saved here')
    parser.add_argument("--songs_dir", type=str, default=SONGS_LIST_DIR, help='directory with songs that you wish to create a playlist out of them')
    parser.add_argument("--outpath", type=str, default=DEFAULT_OUT_PATH, help='the output playlist and numpy arrays will be saved here')
    parser.add_argument("--fade_duration", type=float, default=FADE_DURATION, help='fade duration. used in playlister_playlist_fader.wav file')
    parser.add_argument("--use_spleeter", action='store_true', default=False,
                        help='whether to use spleeter and choose the transition point based only on the accompaniment or not')

    args = parser.parse_args()
    os.environ['HF_HOME'] = args.home_dir

    create_full_playlist_using_dtw(args.outpath, args.songs_dir, home_directory=args.home_dir,
                                   use_spleeter=args.use_spleeter, fade_duration=args.fade_duration)

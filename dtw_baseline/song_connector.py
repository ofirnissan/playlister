from song_handler import Song
import soundfile as sf
import librosa
import numpy as np
import sys
import os
sys.path.append("audio_hw2_code_only")
from plotter import Plotter
from spleeter.separator import Separator


# songs_list_dir = '/home/yandex/APDL2324a/group_7/haviv_playlist/'
songs_list_dir = '/mnt/c/Users/ofirn/Music/songs/'
no_vocal_detection_threshold = 5 # dB ; Threshold for silence detection in the vocals energy array. 
noise_threshold_for_transformation = 0  # dB ; Threshold for silence detection in transformation.
min_size_of_time_interval = 2 # sec ; Minimum size of time interval for silence detection.
jump = 2 # Minimum number of frames between two silent intervals to consider them as different intervals. 
          # This is used to deal with noise in the energy array of the vocals.
fade_duration = 2.0 # sec ; Duration of the fade in and fade out effects.


# =================================================================================
# A Python3 program for 
# Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix 
# representation of the graph

# Library for INT_MAX
import sys


class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                    for row in range(vertices)]

    # A utility function to print 
    # the constructed MST stored in parent[]
    def printAndGetMST(self, parent):
        mst = {}
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            if parent[i] not in mst:
                mst[parent[i]] = []
            mst[parent[i]].append(i)
        return mst

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False \
                and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        return self.printAndGetMST(parent)
        

# =================================================================================

# Contributed by Divyanshu Mehta (https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/)



def get_partial_audio(audio, sr, start_sec=None, end_sec=None):
    start_index = int(sr * start_sec) if start_sec is not None else 0
    end_index = int(sr * end_sec) if end_sec is not None else len(audio)
    return audio[start_index: end_index]


def connect_between_songs_by_dtw_only(song1: Song, song2: Song, file_path: str, song_builder: Song=None, accompaniment=False, pick_idx_method='min_dist_in_path'):
    print(f"Concat {songs[i].song_name} and {songs[i+1].song_name} using DTW method (over chroma stft of the songs)")
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
    window_in_sec = int(5 * song1.sr / 512) 
    dtw_cost_matrix, wp = librosa.sequence.dtw(suffix_chroma_stft, prefix_chroma_stft, subseq=True)  # wp is the Warping path
    if pick_idx_method == 'random':
        sorted_indeces = np.random.choice(len(wp), len(wp))
    elif pick_idx_method == 'max_propagation':
        propagation = np.cumsum(np.sum(wp[:-1] - wp[1:], axis=1))
        sorted_indeces = np.argsort(propagation[window_in_sec:] - propagation[:len(propagation) - window_in_sec])[::-1] 
    elif pick_idx_method == 'min_dist_in_path':
        path_cost = dtw_cost_matrix[wp[:, 0], wp[:, 1]]
        sorted_indeces = np.argsort(path_cost[:len(path_cost) - window_in_sec] - path_cost[window_in_sec:])
    
    # Among chosen indeces by the above methods, choose index in which suffix is not scilent:
    first_audio = None
    for idx in sorted_indeces:
        idx += window_in_sec  # wp start from max to min. So by increasing idx by window, we get the beginning of the match
        suffix_cut_frame, prefix_cut_frame = wp[idx]
        first_song = song_builder if song_builder is not None else song1
        prefix_cut_audio_index = prefix_cut_frame * 512
        suffix_cut_audio_index = len(first_song.audio) - (len(suffix_audio) - suffix_cut_frame * 512)
        first_audio = first_song.audio[: suffix_cut_audio_index]
        second_audio = song2.audio[prefix_cut_audio_index:]
        if first_song.get_audio_energy_array(get_partial_audio(first_audio, first_song.sr, start_sec=-2))[0].mean() > noise_threshold_for_transformation:
            break

    # Concat songs with overlapping fade and save new song:
    new_audio = fadeout_cur_fadein_next(first_audio, second_audio, song1.sr, duration=fade_duration)
    sf.write(file_path, new_audio, first_song.sr)
    new_song = Song(file_path)
    return new_song


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
                if song1.get_audio_energy_array(get_partial_audio(first_audio, song1.sr, start_sec=-2))[0].mean() > noise_threshold_for_transformation:
                    break
            if distances_in_path[idx] < min_value:
                min_value = distances_in_path[idx]
                min_idx = idx
                min_i = i
                min_j = j
    return min_i, min_j, wp[min_idx]


def concat_between_songs_post_spleeter(song1: Song, song2: Song, file_path: str, song_builder: Song = None):
    print(f"Concat {song1.song_name} and {song2.song_name} using vocals and accompaniment separation then dtw")
    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    
    # Get scilent time intervals of prefix and suffix of both songs:
    song1_prefix_scilent_intervals, song1_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song1, threshold=no_vocal_detection_threshold, min_time_interval_length=min_size_of_time_interval)
    song2_prefix_scilent_intervals, song2_suffix_scilent_intervals = find_silent_intervals_of_suffix_and_prefix(\
        song2, threshold=no_vocal_detection_threshold, min_time_interval_length=min_size_of_time_interval)
    
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


def organize_song_list_using_dtw_optimum_aproximation(songs):
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
    mst = g.primMST()
    # create the organized playlist using pre-order traversal of the mst
    organized_songs = []
    def pre_order_traversal(mst, node):
        organized_songs.append(node)
        if node in mst:
            for child in mst[node]:
                pre_order_traversal(mst, child)
    pre_order_traversal(mst, 0)
    return [songs[i] for i in organized_songs]
    

def organize_song_list_using_tempo(songs):
    # Sort the songs by tempo
    songs = sorted(songs, key=lambda x: x.tempo)
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
    song_names = os.listdir(songs_list_dir)[:3]
    songs = []
    # get all files from dir to song_names list:
    # sep = Separator('spleeter:2stems')
    sep = None
    for song_name in song_names:
        print(f"parsing: {song_name}")
        song = Song(songs_list_dir + f"{song_name}", seperator=sep, remove_zero_amp=True)
        song.partial_audio_time_in_sec = 30
        time_in_sec = song.partial_audio_time_in_sec
        # song.calc_tempo()
        # song.find_vocals_and_accompaniment_for_suffix_and_prefix(dir_path=f"/mnt/c/Users/ofirn/Documents/oni/elec/playlister/spleeter_output/{song_name}")
        # song.get_prefix_and_suffix_energy_array_post_separation()
        songs.append(song)
        
        # song.suffix_vocals.plotter.plot_energy_and_rms(threshold=no_vocal_detection_threshold, plot_rms=True, title="suffix_vocal")
        # song.prefix_vocals.plotter.plot_energy_and_rms(threshold=no_vocal_detection_threshold, plot_rms=True, title="prefix_vocal")

        # suffix_song.plotter.plot_mel_spectogram()

    # songs = organize_song_list_using_tempo(songs)
    # songs = organize_song_list_using_dtw_optimum_aproximation(songs)
    songs = organize_song_list_using_dtw_greedy(songs)
    print([song.song_name for song in songs])
    new_song = songs[0]
    print('concatinating songs...')
    result_fp = 'concat_song.wav'
    for i in range(len(songs)-1):
        # new_song = concat_between_songs_post_spleeter(songs[i], songs[i+1], file_path=result_fp, song_builder=new_song)
        new_song = connect_between_songs_by_dtw_only(songs[i], songs[i+1], file_path=result_fp, song_builder=new_song)
        # new_song = connect_between_songs_by_dtw_only(songs[i], songs[i+1], file_path=result_fp, song_builder=new_song, accompaniment=True)
    sf.write(result_fp, new_song.audio, new_song.sr)

    

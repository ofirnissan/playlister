import soundfile as sf
import sys
import numpy as np
import os

FADE_DURATION = 3


def save_audio_file(file_path, audio, sr):
    num_retries = 3
    while num_retries > 0:
        try:
            sf.write(file_path, audio, sr)
            break
        except Exception as e:
            num_retries -= 1
            print(f"Failed to save the audio file, number of retries left: {num_retries}")
    if num_retries == 0:
        print(f"Failed")

# =================================================================================
# A Python3 program for
# Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix
# representation of the graph

# Library for INT_MAX

class Graph:
    # Written by Divyanshu Mehta (https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/)
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
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

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

    def find_approximate_optimal_tsp_path(self):
        mst = self.primMST()
        # create the organized playlist using pre-order traversal of the mst
        organized_vertex = []

        def pre_order_traversal(mst, node):
            organized_vertex.append(node)
            if node in mst:
                for child in mst[node]:
                    pre_order_traversal(mst, child)
            return organized_vertex

        pre_order_traversal(mst, 0)
        return organized_vertex


def fadeout_cur_fadein_next(audio1, audio2, sr, duration=FADE_DURATION, overlap=True):
    new_audio_1 = apply_fadeout(audio1, sr, duration)
    new_audio_2 = apply_fadein(audio2, sr, duration)
    length = int(duration*sr)
    if overlap:
        new_audio = new_audio_1[:]
        new_audio[-length:] += new_audio_2[:length]
        new_audio = np.concatenate((new_audio, new_audio_2[length:]))
        return new_audio
    else:
        new_audio = np.concatenate((new_audio_1, new_audio_2))
        return new_audio


def apply_fadeout(audio, sr, duration=FADE_DURATION):
    length = int(duration*sr)
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)
    # apply the curve
    new_audio = audio[:]
    new_audio[-length:] = new_audio[-length:] * fade_curve
    return new_audio


def apply_fadein(audio, sr, duration=FADE_DURATION):
    length = int(duration*sr)
    # linear fade
    fade_curve = np.linspace(0.0, 1.0, length)
    # apply the curve
    new_audio = audio[:]
    new_audio[:length] = new_audio[:length] * fade_curve
    return new_audio


def get_partial_audio(audio, sr, start_sec=None, end_sec=None):
    start_index = int(sr * start_sec) if start_sec is not None else 0
    end_index = int(sr * end_sec) if end_sec is not None else len(audio)
    return audio[start_index: end_index]


def show_dtw_cost_matrix_and_wp(dtw_cost_matrix, wp):
    import matplotlib.pyplot as plt
    import librosa
    fig, ax = plt.subplots()
    img = librosa.display.specshow(dtw_cost_matrix, x_axis='frames', y_axis='frames', ax=ax, hop_length=512)
    ax.set(title='DTW cost', xlabel='prefix', ylabel='sufix')
    ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax.legend()
    fig.colorbar(img, ax=ax)
    plt.show()


def songs_spleeter(songs_list, time_in_sec, spleeter_output_dir_path):
    from spleeter.separator import Separator
    sep = Separator('spleeter:2stems')
    for song in songs_list:
        song.seperator = sep  # set the seperator
        song.partial_audio_time_in_sec = time_in_sec  # set the partial audio time in sec of suffix and prefix
        song_spleeter_out_path = os.path.join(spleeter_output_dir_path, song.song_name) # set the output path
        if not os.path.exists(song_spleeter_out_path): # create the output path if not exists
            os.mkdir(song_spleeter_out_path)
        # find the vocals and accompaniment for suffix and prefix and save them to the appropriate attributes
        song.find_vocals_and_accompaniment_for_suffix_and_prefix(dir_path=song_spleeter_out_path) 
        # get the energy array for the suffix and prefix and save them to the appropriate attributes
        song.get_prefix_and_suffix_energy_array_post_separation()

    return # no need to return anything since the songs_list is passed by reference
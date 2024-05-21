import numpy as np
import matplotlib.pyplot as plt
import os
from song_handler import Song

FULL_WINDOW_SECONDS = 45
FADE_DURATION = 2.0
PLAYLIST = "yael_playlist"
songs_files_dir = "/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/"
np_files_dir = "/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/playlister_project/playlister_outputs_ofir/"
dtw_np_files_path = "/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/outputs/yael_playlist/dtw/"
window_pairs = [(200, 50), (200, 100), (250, 50), (250, 100), (250, 150)]
playlists_dict = {}

for window_size_samples_suffix, window_size_samples_prefix in window_pairs:
    np_files_path = os.path.join(np_files_dir, f"{PLAYLIST}/s_{int(window_size_samples_suffix*0.02)}_p_{int(window_size_samples_prefix*0.02)}")
    songs_name_order = np.load(os.path.join(np_files_path, "songs_name_order.npy"))
    music_dir = os.path.join(songs_files_dir, PLAYLIST)
    songs_file_names_list = os.listdir(music_dir)

    order_songs_list = []
    for song_prefix_name in songs_name_order:
        for song_name in songs_file_names_list:
            if song_prefix_name == song_name.split(".")[0]:
                song = Song(os.path.join(music_dir, song_name))
                song.calc_tempo()
                order_songs_list.append(song)
                break
    playlists_dict[(window_size_samples_suffix, window_size_samples_prefix)] = order_songs_list


# calculate DTW soulution tempo trajectories
baseline_song_name_order = np.load(os.path.join(dtw_np_files_path, "baseline_songs_name_order.npy"))
order_songs_list_dtw = []
for song_prefix_name in baseline_song_name_order:
    for song_name in songs_file_names_list:
        if song_prefix_name == song_name.split(".")[0]:
            song = Song(os.path.join(music_dir, song_name))
            song.calc_tempo()
            order_songs_list_dtw.append(song)
            break
playlists_dict["dtw"] = order_songs_list_dtw


# check if there are equal tempo values for entire playlist and if so update the key to contain all the playlists with the same tempo list
for key1 in playlists_dict.keys():
    for key2 in playlists_dict.keys():
        if key1 != key2 and [song.tempo for song in playlists_dict[key1]] == [song.tempo for song in playlists_dict[key2]]:
            playlists_dict[key1] = (playlists_dict[key1], playlists_dict[key2])
            del playlists_dict[key2]

# plot all the tempo trajectories for all the playlists on the same graph
plt.figure()
for key in playlists_dict.keys():
    plt.plot([song.tempo for song in playlists_dict[key]], label=str(key))
plt.title("Tempo trajectories")
plt.ylabel("Tempo")
plt.xlabel("Song index")
plt.legend()
plt.savefig("tempo_trajectories.png")


for key in playlists_dict.keys():
    print(key)
    for song in playlists_dict[key]:
        print(song.tempo)
    print("\n")
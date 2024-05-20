from song_handler import Song
import soundfile as sf
import librosa
import numpy as np
import os
from utils import Graph, fadeout_cur_fadein_next, get_partial_audio
from utils import show_dtw_cost_matrix_and_wp  # for debug

FULL_WINDOW_SECONDS = 45
WINDOW_SIZE_SAMPLES_PREFIX = 100
WINDOW_SIZE_SAMPLES_SUFFIX = 200
FADE_DURATION = 2.0

np_files_path = "/mnt/c/Users/ofirn/Downloads/np/playlister_outputs"
cut_indices_prefix = np.load(os.path.join(np_files_path, "cut_indices_prefix_window_100_hop_25.npy")) 
cut_indices_suffix = np.load(os.path.join(np_files_path, "cut_indices_suffix_window_200_hop_25.npy"))
playlister_playlist_numpy = np.load(os.path.join(np_files_path, "playlister_playlist_numpy.npy"))
playlister_playlist_fader_numpy = np.load(os.path.join(np_files_path, "playlister_playlist_fader_numpy.npy"))
songs_name_order = np.load(os.path.join(np_files_path, "songs_name_order.npy"))
adjecency_matrix = np.load(os.path.join(np_files_path, "adjacency_matrix.npy"))
music_dir = "/mnt/c/Users/ofirn/Music/songs"
songs_file_names_list = os.listdir(music_dir)

order_songs_list = []
for song_prefix_name in songs_name_order:
    for song_name in songs_file_names_list:
        if song_prefix_name == song_name.split(".")[0]:
            song = Song(os.path.join(music_dir, song_name))
            order_songs_list.append(song)
            break

print([song.song_name for song in order_songs_list])


number_of_songs = len(order_songs_list)
songs_time_len_after_cut = []

for i in range(number_of_songs):
    start_sec = 0 if i == 0 else cut_indices_prefix[[i-1], [i]]*0.02
    end_sec = 0 if i == number_of_songs-1 else - FULL_WINDOW_SECONDS + \
                                                    (cut_indices_suffix[[i], [i+1]] +
                                                    WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02
    end_sec = len(order_songs_list[i].audio)/order_songs_list[i].sr + end_sec
    song_time_after_cut = end_sec - start_sec
    songs_time_len_after_cut.append(song_time_after_cut)

full_audio_transition_times = np.cumsum(songs_time_len_after_cut)
full_audio_transition_times = full_audio_transition_times/60

print(full_audio_transition_times)

    # curr_song_partial_audio = order_songs_list[i].get_partial_audio(start_sec=start_sec, end_sec=end_sec)
    # full_playlist_audio = np.concatenate([full_playlist_audio, curr_song_partial_audio])
    # if i != 0:
    #     full_playlist_audio_fader = fadeout_cur_fadein_next(full_playlist_audio_fader, curr_song_partial_audio,
    #                                                         32000, duration=FADE_DURATION)
    # else:
    #     full_playlist_audio_fader = curr_song_partial_audio

    # try:
    #     np.save(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen_spleeter/playlister_playlist_numpy.npy', full_playlist_audio)
    #     np.save(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen_spleeter/playlister_playlist_fader_numpy.npy', full_playlist_audio_fader)
    # except Exception as e:
    #     print("Failed to save the numpy arrays")

    # save_audio_file(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen_spleeter/playlister_playlist.wav', full_playlist_audio, songs_list[0].sr)
    # save_audio_file(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen_spleeter/playlister_playlist_fader.wav', full_playlist_audio_fader, songs_list[0].sr)


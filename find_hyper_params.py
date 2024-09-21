from song_handler import Song
import torch
import numpy as np
import os
from musicgen_based_playlist_generator import NUMBER_OF_CODEBOOKS, calculate_log_prob_of_sequence_given_another_sequence
import matplotlib.pyplot as plt


DEFAULT_SONGS_DIR = 'yael_playlist'
PATH = "/home/joberant/NLP_2324/yaelshemesh"

def calculate_loss(songs_tokens, suffix_window_size_sec, prefix_window_size_sec, model, codebook=0):
    suffix_window_size_tokens = int(suffix_window_size_sec / 0.02)
    prefix_window_size_tokens = int(prefix_window_size_sec / 0.02)
    probabilities_for_same_song_all = 0
    probabilities_for_another_song_all = 0
    for i, song_tokens in enumerate(songs_tokens):
        print(f"Song #{i}")
        list_of_song_suffixes = torch.from_numpy(np.array([song_tokens.audio_codes[..., k:k+suffix_window_size_tokens][0, 0]
                                              for k in range(0, song_tokens.audio_codes.shape[-1] - prefix_window_size_tokens - suffix_window_size_tokens,
                                                             suffix_window_size_tokens)]))
        list_of_song_suffixes = list_of_song_suffixes.reshape(list_of_song_suffixes.shape[0] *
                                                              list_of_song_suffixes.shape[1],
                                                              list_of_song_suffixes.shape[2])
        list_of_song_prefixes = torch.from_numpy(np.array([song_tokens.audio_codes[..., k:k+prefix_window_size_tokens][0, 0]
                                              for k in range(suffix_window_size_tokens, song_tokens.audio_codes.shape[-1] - prefix_window_size_tokens,
                                                             suffix_window_size_tokens)]))
        list_of_song_prefixes = list_of_song_prefixes.reshape(list_of_song_prefixes.shape[0] *
                                                              list_of_song_prefixes.shape[1],
                                                              list_of_song_prefixes.shape[2])

        list_of_prefixes_from_another_song = []
        for j, another_song_tokens in enumerate(songs_tokens):
            if j == i:
                continue
            list_of_prefixes_from_another_song += [another_song_tokens.audio_codes[..., k:k+prefix_window_size_tokens][0, 0]
                                                   for k in range(suffix_window_size_tokens,
                                                                  another_song_tokens.audio_codes.shape[-1] - prefix_window_size_tokens,
                                                                  suffix_window_size_tokens)]

        np.random.shuffle(list_of_prefixes_from_another_song)  # Ofir: why do you need shuffle?

        list_of_prefixes_from_another_song_rand = list_of_prefixes_from_another_song[:len(list_of_song_suffixes) // NUMBER_OF_CODEBOOKS]
        list_of_prefixes_from_another_song_rand = torch.from_numpy(np.array(list_of_prefixes_from_another_song_rand))
        list_of_prefixes_from_another_song_rand = list_of_prefixes_from_another_song_rand.reshape(
            list_of_prefixes_from_another_song_rand.shape[0] * list_of_prefixes_from_another_song_rand.shape[1],
            list_of_prefixes_from_another_song_rand.shape[2])

        probabilities_for_same_song = calculate_log_prob_of_sequence_given_another_sequence(list_of_song_suffixes,
                                                                                            list_of_song_prefixes,
                                                                                            model, codebook=codebook)
        probabilities_for_another_song = calculate_log_prob_of_sequence_given_another_sequence(
            list_of_song_suffixes, list_of_prefixes_from_another_song_rand, model, codebook=codebook)
        probabilities_for_same_song_all += sum(probabilities_for_same_song) / len(probabilities_for_same_song)
        probabilities_for_another_song_all += sum(probabilities_for_another_song) / len(probabilities_for_another_song)
    return probabilities_for_same_song_all, probabilities_for_another_song_all


def plot_loss(title, x_label, x, y):
    plt.figure()
    plt.plot(x, y, s=50)
    plt.xlabel(x_label)
    plt.ylabel('loss')
    plt.title(title)
    plt.show()


def find_hyper_params(songs_dir, home_dir):
    global audio_encoder
    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(3)]
    from transformers import MusicgenForConditionalGeneration
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", cache_dir=home_dir)
    audio_encoder = model.audio_encoder
    songs_tokens = []
    for song in songs_list:
        curr_tokens = audio_encoder.encode(torch.from_numpy(song.audio.reshape(1, 1, len(song.audio))))
        songs_tokens.append(curr_tokens)
    for hyperparams_list in [
        # different codebooks
        [(5, 2, 0), (5, 2, 1), (5, 2, 2), (5, 2, 3)],
        # different prefix_window_size_sec
        [(4, 0.02, 0), (4, 0.25, 0), (4, 0.5, 0), (4, 1, 0), (4, 1.5, 0), (4, 2, 0), (4, 2.5, 0), (4, 3, 0), (4, 4, 0)],
        # different suffix_window_size_sec
        [(2, 2, 0), (2.5, 2, 0), (3, 2, 0), (3.5, 2, 0), (4, 2, 0), (4.5, 2, 0), (5, 2, 0), (6, 2, 0), (7, 2, 0)]
    ]:
        for suffix_window_size_sec, prefix_window_size_sec, codebook in hyperparams_list:
            print(
                f"suffix_window_size_sec: {suffix_window_size_sec}; prefix_window_size_sec: {prefix_window_size_sec}; codebook: {codebook}")
            probabilities_for_same_song, probabilities_for_another_song = calculate_loss(songs_tokens,
                                                                                         suffix_window_size_sec,
                                                                                         prefix_window_size_sec, model,
                                                                                         codebook=codebook)
            loss = probabilities_for_another_song - probabilities_for_same_song
            print(
                f"probabilities_for_same_song: {probabilities_for_same_song}, probabilities_for_another_song: {probabilities_for_another_song}, Loss: {loss}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Characterize hyper params to find the best fit.')
    parser.add_argument("--home_dir", type=str, default=PATH, help='model cache will be saved here')
    parser.add_argument("--songs_dir", type=str, default=DEFAULT_SONGS_DIR,
                        help='directory with songs that you wish to create a playlist out of them')

    args = parser.parse_args()
    os.environ['HF_HOME'] = args.home_dir

    find_hyper_params(args.songs_dir, args.home_dir)

    #  ------------- plot example --------------

    # suffix_x = [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7]
    # suffix_y = [-179.7879638671875, -192.915771484375, -214.9764404296875, -217.2119140625, -244.0914306640625,
    #             -251.3134765625, -269.43701171875, -273.569091796875, -266.1756591796875]
    # plot_loss('Loss for Prefix Window: 2 Sec and Different Suffix Windows', 'suffix window (sec)', suffix_x, suffix_y)




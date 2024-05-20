from typing import List
from song_handler import Song
import torch
import numpy as np
import os
from song_connector_musicgen import NUMBER_OF_CODEBOOKS, calculate_log_prob_of_sequence_given_another_sequence


SONG_DIRECTORY = 'C:\\Users\\yaelshe\\PycharmProjects\\playlister\\yael_playlist'


def calculate_loss(songs_tokens, suffix_window_size_sec, prefix_window_size_sec, model):
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
                                                                                            model)
        probabilities_for_another_song = calculate_log_prob_of_sequence_given_another_sequence(
            list_of_song_suffixes, list_of_prefixes_from_another_song_rand, model)
        probabilities_for_same_song_all += sum(probabilities_for_same_song) / len(probabilities_for_same_song)
        probabilities_for_another_song_all += sum(probabilities_for_another_song) / len(probabilities_for_another_song)
    return probabilities_for_same_song_all, probabilities_for_another_song_all


if __name__ == '__main__':
    songs_dir = 'playlist'
    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(3)]

    from transformers import MusicgenForConditionalGeneration
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    audio_encoder = model.audio_encoder
    songs_tokens = []
    for song in songs_list:
        curr_tokens = audio_encoder.encode(torch.from_numpy(song.audio.reshape(1, 1, len(song.audio))))
        songs_tokens.append(curr_tokens)

    for suffix_window_size_sec, prefix_window_size_sec in [(2, 2), (2.5, 2), (3, 2), (3.5, 2), (4, 2), (4.5, 2), (5, 2), (6, 2), (7, 2)]:
        print(f"suffix_window_size_sec: {suffix_window_size_sec}; prefix_window_size_sec: {prefix_window_size_sec}")
        probabilities_for_same_song, probabilities_for_another_song = calculate_loss(songs_tokens, suffix_window_size_sec, prefix_window_size_sec, model)
        loss = probabilities_for_another_song - probabilities_for_same_song
        print(f"probabilities_for_same_song: {probabilities_for_same_song}, probabilities_for_another_song: {probabilities_for_another_song}, Loss: {loss}")

import os

import numpy as np
from dtw_baseline.song_handler import Song
import torch
import sys
from utils import Graph, fadeout_cur_fadein_next, save_audio_file

FADE_DURATION = 3.0
NUMBER_OF_CODEBOOKS = 4

HOP_SIZE_SAMPLES = 50  # 1 sec / 0.02
WINDOW_SIZE_SAMPLES_SUFFIX = 200  # 4 sec/ 0.02
WINDOW_SIZE_SAMPLES_PREFIX = 100  # 2 sec/ 0.02

FULL_WINDOW_SECONDS = 30


def calculate_log_prob_of_sequence_given_another_sequence(token_sequence_1, token_sequence_2, model, text_tokens):
    tokens = torch.cat([token_sequence_1, token_sequence_2], dim=-1)

    text_tokens = torch.tile(text_tokens, (tokens.shape[0]//NUMBER_OF_CODEBOOKS, 1))
    with torch.no_grad():
        outputs = model(input_ids=text_tokens, decoder_input_ids=tokens)
        logits = outputs.logits

    sequence_2_logits = logits[:, -token_sequence_2.shape[1] - 1:-1]
    sequence_2_logmax = torch.nn.functional.log_softmax(sequence_2_logits, dim=-1)

    # get the probability for the specific sequence
    sequence_2_logmax = sequence_2_logmax.reshape((token_sequence_2.shape[0]*token_sequence_2.shape[1], 2048))[
        range(token_sequence_2.shape[0]*token_sequence_2.shape[1]),
        token_sequence_2.reshape(token_sequence_2.shape[0]*token_sequence_2.shape[1])].reshape(token_sequence_2.shape[0],token_sequence_2.shape[1])

    batch_sequence_2_logmax = sequence_2_logmax[range(0, logits.shape[0], NUMBER_OF_CODEBOOKS)]

    return torch.sum(batch_sequence_2_logmax, dim=-1)


def calculate_log_prob_of_sequence_given_another_sequence_method_2(token_sequence_1, token_sequence_2, model, text_tokens):
    tokens = torch.cat([token_sequence_1, token_sequence_2], dim=-1)

    log_sum = 0
    # loop for every token in prefix
    for i in range(0, WINDOW_SIZE_SAMPLES_PREFIX):
        curr_tokens = tokens[..., 0: WINDOW_SIZE_SAMPLES_SUFFIX + i]
        with torch.no_grad():
            outputs = model(input_ids=text_tokens, decoder_input_ids=curr_tokens)
            logits = outputs.logits
        past_tok, current_tok = i, i + 1
        log_token_prob = 0
        for token_dim_index in range(1):
            token_logit = logits[token_dim_index, -1, :]
            token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
            log_token_prob += token_log_probs[tokens[token_dim_index, current_tok]].item()

        log_sum += log_token_prob
        # print(f"Token, Log Prob: {log_token_prob}")

    return log_sum


def connect_between_songs(song1: Song, song2: Song):
    assert song1.sr == song2.sr
    from transformers import AutoTokenizer, MusicgenForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    audio_encoder = model.audio_encoder
    text_tokens = [tokenizer.pad_token_id]
    text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))

    suffix = song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS)
    prefix = song2.get_partial_audio(end_sec=FULL_WINDOW_SECONDS)

    prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
    suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))


    tuples = []
    partial_suffix_tokens_batched = torch.tensor([], dtype=torch.int)
    partial_prefix_tokens_batched = torch.tensor([], dtype=torch.int)

    for i1 in range(0, suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, HOP_SIZE_SAMPLES):
        for i2 in range(0, prefix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_PREFIX, HOP_SIZE_SAMPLES):
            print(f'{i1, i2} / {suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, prefix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_PREFIX}')

            transition_energy, _ = song1.get_audio_energy_array(
                song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS + (i1 + WINDOW_SIZE_SAMPLES_SUFFIX)*0.02 - 1,
                                        end_sec=-FULL_WINDOW_SECONDS + (i1 + WINDOW_SIZE_SAMPLES_SUFFIX)*0.02))
            if np.mean(transition_energy) < 0:
                continue

            transition_energy, _ = song2.get_audio_energy_array(
                song2.get_partial_audio(start_sec=i2 * 0.02, end_sec=i2 * 0.02 + 1))
            if np.mean(transition_energy) < 0:
                continue
            tuples.append((i1, i2))
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i1:i1+WINDOW_SIZE_SAMPLES_SUFFIX][0, 0]
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i2:i2+WINDOW_SIZE_SAMPLES_PREFIX][0, 0]

            partial_suffix_tokens_batched = torch.cat([partial_suffix_tokens_batched, partial_suffix_tokens], dim=0)
            partial_prefix_tokens_batched = torch.cat([partial_prefix_tokens_batched, partial_prefix_tokens], dim=0)

    log_sum = calculate_log_prob_of_sequence_given_another_sequence(partial_suffix_tokens_batched,
                                                                    partial_prefix_tokens_batched, model, text_tokens)

    best_prob = torch.max(log_sum)
    print(f"Total Log Sum Probability: {best_prob}")
    best_tuple = tuples[torch.argmax(log_sum)]
    print(f"Best tuple: {best_tuple}")

    concat_audio = np.concatenate(
        [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02),
         song2.get_partial_audio(start_sec=best_tuple[1] * 0.02)])
    save_audio_file(f'{song1.song_name} + {song2.song_name}_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)
    save_audio_file(f'{song1.song_name} + {song2.song_name}_no_fader.wav', concat_audio, song1.sr)

    concat_audio = np.concatenate(
        [song1.get_partial_audio(start_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + best_tuple[0] * 0.02,
                                 end_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02),
         song2.get_partial_audio(start_sec=best_tuple[1] * 0.02,
                                 end_sec=best_tuple[1] * 0.02 + WINDOW_SIZE_SAMPLES_PREFIX * 0.02)])
    save_audio_file(f'{song1.song_name} + {song2.song_name}_partial_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)

    print(f'Best indices: {best_tuple}')

    concat_audio = fadeout_cur_fadein_next(
        song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02, len(song1.audio))),
        song2.get_partial_audio(start_sec=best_tuple[1]*0.02), song1.sr,
        duration=FADE_DURATION)

    save_audio_file(f'{song1.song_name} + {song2.song_name}_long_fader.wav', concat_audio, song1.sr)

    concat_audio = fadeout_cur_fadein_next(
        song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02, len(song1.audio))),
        song2.get_partial_audio(start_sec=best_tuple[1]*0.02), song1.sr,
        duration=1)

    save_audio_file(f'{song1.song_name} + {song2.song_name}_short_fader.wav', concat_audio, song1.sr)

    return best_prob, best_tuple


def create_full_playlist(songs_dir):
    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(number_of_songs)]
    adjacency_matrix = np.ones((number_of_songs, number_of_songs)) * np.inf
    cut_indices_suffix = np.zeros((number_of_songs, number_of_songs))
    cut_indices_prefix = np.zeros((number_of_songs, number_of_songs))
    for i in range(number_of_songs):
        song1 = songs_list[i]
        for j in range(number_of_songs):
            if i == j:
                continue
            song2 = songs_list[j]
            best_prob, best_tuple = connect_between_songs(song1, song2)
            adjacency_matrix[i, j] = best_prob
            cut_indices_suffix[i, j] = best_tuple[0]
            cut_indices_prefix[i, j] = best_tuple[1]

    # the log probabilities are negative, and we what to choose the highest probabilities, so in order to use the tsp
    # approximation algorithm, we converted it to positive value
    adjacency_matrix = adjacency_matrix * -1

    g = Graph(number_of_songs)
    g.graph = adjacency_matrix

    organized_songs = g.find_approximate_optimal_tsp_path()
    songs_order = [songs_list[i].song_name for i in organized_songs]

    full_playlist_audio = []
    # create the playlist
    for i in range(number_of_songs):
        start_sec = 0 if i == 0 else cut_indices_prefix[songs_order[i-1], songs_order[i]]*0.02
        end_sec = None if i == number_of_songs-1 else - FULL_WINDOW_SECONDS + \
                                                      (cut_indices_suffix[songs_order[i], songs_order[i+1]] +
                                                       WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02
        curr_song_partial_audio = songs_list[songs_order[i]].get_partial_audio(start_sec=start_sec, end_sec=end_sec)
        full_playlist_audio += curr_song_partial_audio

    save_audio_file(f'playlister_playlist.wav', full_playlist_audio, songs_list[0].sr)


if __name__ == '__main__':

    create_full_playlist('../eyal - part')
    # song1 = Song(f"../eyal/yafyufa.mp3", sr=32000)
    # song2 = Song(f"../eyal/malkat hayofi.mp3", sr=32000)
    # connect_between_songs(song1, song2)
    #
    # songs_pairs = np.array([[(song_name_1, song_name_2) for song_name_1 in os.listdir("../eyal")] for song_name_2 in os.listdir("../eyal")])
    # songs_pairs = songs_pairs.reshape((songs_pairs.shape[0] * songs_pairs.shape[1], 2))
    # np.random.shuffle(songs_pairs)
    #
    # for song_name_1, song_name_2 in songs_pairs:
    #     print(song_name_1, song_name_2)
    #     if song_name_1 == song_name_2:
    #         continue
    #     try:
    #         song1 = Song(f"../eyal/{song_name_1}", sr=32000)
    #         song2 = Song(f"../eyal/{song_name_2}", sr=32000)
    #
    #         # song1 = Song("../songs/Wish You Were Here - Incubus - Lyrics.mp3", sr=32000)
    #         # song2 = Song("../songs/Incubus - Drive.mp3", sr=32000)
    #
    #     except Exception as e:
    #         continue
    #     connect_between_songs(song1, song2)



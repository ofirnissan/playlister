import os

import numpy as np
from song_handler import Song
import torch
from utils import Graph, fadeout_cur_fadein_next, save_audio_file
from transformers import AutoTokenizer, MusicgenForConditionalGeneration

os.environ['HF_HOME'] = '/home/joberant/NLP_2324/yaelshemesh'

DEVICE = 'cuda:3'

FADE_DURATION = 2.0
NUMBER_OF_CODEBOOKS = 4

HOP_SIZE_SAMPLES = 25  # 0.5 sec / 0.02
WINDOW_SIZE_SAMPLES_SUFFIX = 200  # 4 sec/ 0.02
WINDOW_SIZE_SAMPLES_PREFIX = 100  # 2 sec/ 0.02

FULL_WINDOW_SECONDS = 45

BATCH_SIZE = 25


def calculate_log_prob_of_sequence_given_another_sequence(token_sequence_1, token_sequence_2, model, text_tokens):
    tokens = torch.cat([token_sequence_1, token_sequence_2], dim=-1)
    text_tokens = torch.tile(text_tokens, (tokens.shape[0]//NUMBER_OF_CODEBOOKS, 1))

    print(f"coda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        model.to(DEVICE)
        tokens = tokens.to(DEVICE)  # Move tokens to GPU
        text_tokens = text_tokens.to(DEVICE)  # Move text_tokens to GPU

    with torch.no_grad():
        outputs = model(input_ids=text_tokens, decoder_input_ids=tokens)
        logits = outputs.logits

    sequence_2_logits = logits[:, -token_sequence_2.shape[1] - 1:-1]
    sequence_2_logmax = torch.nn.functional.log_softmax(sequence_2_logits, dim=-1)

    # get the probability for the specific sequence
    sequence_2_logmax = sequence_2_logmax.reshape((token_sequence_2.shape[0]*token_sequence_2.shape[1], 2048))[
        range(token_sequence_2.shape[0]*token_sequence_2.shape[1]),
        token_sequence_2.reshape(token_sequence_2.shape[0]*token_sequence_2.shape[1])].reshape(token_sequence_2.shape[0],token_sequence_2.shape[1])

    # use only the first token - TODO: try using  all tokens
    batch_sequence_2_logmax = sequence_2_logmax[range(0, logits.shape[0], NUMBER_OF_CODEBOOKS)]

    return torch.sum(batch_sequence_2_logmax, dim=-1)


def calculate_log_prob_of_sequence_given_another_sequence_method_2(token_sequence_1, token_sequence_2, model, text_tokens):
    total_log_sum = torch.zeros((token_sequence_1.shape[0]))
    if torch.cuda.is_available():
        model.to(DEVICE)
    for i in range(token_sequence_2.shape[-1]):
        tokens = torch.cat([token_sequence_1, token_sequence_2[..., 0:i]], dim=-1)
        if torch.cuda.is_available():
            tokens = tokens.to(DEVICE)  # Move tokens to GPU
            text_tokens = text_tokens.to(DEVICE)  # Move text_tokens to GPU

        with torch.no_grad():
            outputs = model(input_ids=text_tokens, decoder_input_ids=tokens)
            logits = outputs.logits

        total_log_sum += get_probability_for_given_token(token_sequence_2[..., i], logits)

    # use only the first token - TODO: try using  all tokens
    total_log_sum = total_log_sum[range(0, total_log_sum.shape[0], NUMBER_OF_CODEBOOKS)]

    return total_log_sum


def get_probability_for_given_token(next_token_batch, logits):
    next_token_logits = logits[..., - 1, :]
    next_token_logits_logmax = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

    # return the probability for the specific sequence
    return next_token_logits_logmax.flatten()[next_token_batch.flatten()].reshape(next_token_batch.shape)


def connect_between_songs(song1: Song, song2: Song):
    assert song1.sr == song2.sr
    import math
    print("connect")
    print("load_model")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    print("encode")
    audio_encoder = model.audio_encoder
    text_tokens = [0]  # pad token id
    text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))

    suffix = song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS)
    prefix = song2.get_partial_audio(end_sec=FULL_WINDOW_SECONDS)

    prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
    suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))

    best_prob = -np.inf
    best_tuple = (suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, 0)

    partial_suffix_tokens_all_batches = torch.tensor([], dtype=torch.int)
    partial_prefix_tokens_all_batches = torch.tensor([], dtype=torch.int)
    tuples = []
    print("create batches")
    for i1 in range(0, suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, HOP_SIZE_SAMPLES):
        print(f'{i1} / {suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX}')
        for i2 in range(0, prefix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_PREFIX, HOP_SIZE_SAMPLES):
            transition_energy, _ = song1.get_audio_energy_array(
                song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS + (i1 + WINDOW_SIZE_SAMPLES_SUFFIX)*0.02 - 1,
                                        end_sec=-FULL_WINDOW_SECONDS + (i1 + WINDOW_SIZE_SAMPLES_SUFFIX)*0.02))
            if np.mean(transition_energy) < 10:
                continue

            transition_energy, _ = song2.get_audio_energy_array(
                song2.get_partial_audio(start_sec=i2 * 0.02, end_sec=i2 * 0.02 + 1))
            if np.mean(transition_energy) < 10:
                continue
            tuples.append((i1, i2))
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i1:i1+WINDOW_SIZE_SAMPLES_SUFFIX][0, 0]
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i2:i2+WINDOW_SIZE_SAMPLES_PREFIX][0, 0]

            partial_suffix_tokens_all_batches = torch.cat([partial_suffix_tokens_all_batches, partial_suffix_tokens], dim=0)
            partial_prefix_tokens_all_batches = torch.cat([partial_prefix_tokens_all_batches, partial_prefix_tokens], dim=0)

    print(f"Number of batches: {partial_prefix_tokens_all_batches.shape[0] / (BATCH_SIZE * 4)}, number of pairs to check: {partial_prefix_tokens_all_batches.shape[0] // NUMBER_OF_CODEBOOKS}")
    print("start magic")
    for batch_number in range(math.ceil(partial_prefix_tokens_all_batches.shape[0]/ (BATCH_SIZE * 4))):
        print(f"Batch #{batch_number}")
        partial_suffix_tokens_batched = partial_suffix_tokens_all_batches[batch_number * BATCH_SIZE * 4: (batch_number + 1) * BATCH_SIZE * 4]
        partial_prefix_tokens_batched = partial_prefix_tokens_all_batches[batch_number * BATCH_SIZE * 4: (batch_number + 1) * BATCH_SIZE * 4]
        tuples_batched = tuples[batch_number * BATCH_SIZE: (batch_number + 1) * BATCH_SIZE]
        log_sum = calculate_log_prob_of_sequence_given_another_sequence(partial_suffix_tokens_batched,
                                                                        partial_prefix_tokens_batched, model,
                                                                        text_tokens)
        cur_best_prob = torch.max(log_sum)
        cur_best_tuple = tuples_batched[torch.argmax(log_sum)]
        if cur_best_prob > best_prob:
            best_prob = cur_best_prob
            best_tuple = cur_best_tuple

    print(f"Total Log Sum Probability: {best_prob}")
    print(f"Best tuple: {best_tuple}")

    # concat_audio = np.concatenate(
    #     [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02),
    #      song2.get_partial_audio(start_sec=best_tuple[1] * 0.02)])
    # save_audio_file(f'{song1.song_name} + {song2.song_name}_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)
    # save_audio_file(f'{song1.song_name} + {song2.song_name}_no_fader.wav', concat_audio, song1.sr)
    #
    # concat_audio = np.concatenate(
    #     [song1.get_partial_audio(start_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + best_tuple[0] * 0.02,
    #                              end_sec=len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02),
    #      song2.get_partial_audio(start_sec=best_tuple[1] * 0.02,
    #                              end_sec=best_tuple[1] * 0.02 + WINDOW_SIZE_SAMPLES_PREFIX * 0.02)])
    # save_audio_file(f'{song1.song_name} + {song2.song_name}_partial_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)

    print(f'Best indices: {best_tuple}')

    # concat_audio = fadeout_cur_fadein_next(
    #     song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02, len(song1.audio))),
    #     song2.get_partial_audio(start_sec=best_tuple[1]*0.02), song1.sr,
    #     duration=FADE_DURATION)
    #
    # save_audio_file(f'{song1.song_name} + {song2.song_name}_long_fader.wav', concat_audio, song1.sr)
    #
    # concat_audio = fadeout_cur_fadein_next(
    #     song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - FULL_WINDOW_SECONDS + (best_tuple[0] + WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02, len(song1.audio))),
    #     song2.get_partial_audio(start_sec=best_tuple[1]*0.02), song1.sr,
    #     duration=1)
    #
    # save_audio_file(f'{song1.song_name} + {song2.song_name}_short_fader.wav', concat_audio, song1.sr)

    return best_prob, best_tuple


def create_full_playlist(songs_dir):
    print("create_play list")
    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(number_of_songs)]
    print("finish load songs")
    adjacency_matrix = np.ones((number_of_songs, number_of_songs)) * np.inf
    cut_indices_suffix = np.zeros((number_of_songs, number_of_songs))
    cut_indices_prefix = np.zeros((number_of_songs, number_of_songs))
    print("start main loop")
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

    organized_songs_indices = g.find_approximate_optimal_tsp_path()

    songs_name_order = [songs_list[i].song_name for i in organized_songs_indices]
    print(songs_name_order)

    full_playlist_audio = []
    full_playlist_audio_fader = []
    # create the playlist
    for i in range(number_of_songs):
        start_sec = 0 if i == 0 else cut_indices_prefix[organized_songs_indices[i-1], organized_songs_indices[i]]*0.02
        end_sec = None if i == number_of_songs-1 else - FULL_WINDOW_SECONDS + \
                                                      (cut_indices_suffix[organized_songs_indices[i], organized_songs_indices[i+1]] +
                                                       WINDOW_SIZE_SAMPLES_SUFFIX) * 0.02
        curr_song_partial_audio = songs_list[organized_songs_indices[i]].get_partial_audio(start_sec=start_sec, end_sec=end_sec)
        full_playlist_audio = np.concatenate([full_playlist_audio, curr_song_partial_audio])
        if i != 0:
            full_playlist_audio_fader = fadeout_cur_fadein_next(full_playlist_audio_fader, curr_song_partial_audio,
                                                                32000, duration=FADE_DURATION)
        else:
            full_playlist_audio_fader = curr_song_partial_audio


    np.save(f'/home/joberant/NLP_2324/yaelshemesh/outputs_concert/playlister_playlist_numpy.npy', full_playlist_audio)
    np.save(f'/home/joberant/NLP_2324/yaelshemesh/outputs_concert/playlister_playlist_fader_numpy.npy', full_playlist_audio_fader)

    save_audio_file(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen/playlister_playlist.wav', full_playlist_audio, songs_list[0].sr)
    save_audio_file(f'/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen/playlister_playlist_fader.wav', full_playlist_audio_fader, songs_list[0].sr)


if __name__ == '__main__':

    create_full_playlist('/home/joberant/NLP_2324/yaelshemesh/haviv')
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



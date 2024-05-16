import os

import numpy as np
from song_handler import Song
import torch
from utils import Graph, fadeout_cur_fadein_next, save_audio_file
from transformers import AutoTokenizer, MusicgenForConditionalGeneration

os.environ['HF_HOME'] = '/home/joberant/NLP_2324/yaelshemesh'

DEVICE = 'cuda:2'

FADE_DURATION = 2.0
NUMBER_OF_CODEBOOKS = 4

HOP_SIZE_SAMPLES = 50  # 0.5 sec / 0.02
WINDOW_SIZE_SAMPLES_SUFFIX = 400  # 4 sec/ 0.02
WINDOW_SIZE_SAMPLES_PREFIX = 200  # 2 sec/ 0.02

FULL_WINDOW_SECONDS = 45

BATCH_SIZE = 25
ENERGY_THRESHOLD = 10


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


def get_transition_indices_list(song: Song, number_of_tokens, suffix=True):
    if suffix:
        window_size = WINDOW_SIZE_SAMPLES_SUFFIX
        energy_start_sec = lambda i: -FULL_WINDOW_SECONDS + (i + window_size) * 0.02 - 1
        energy_end_sec = lambda i: -FULL_WINDOW_SECONDS + (i + window_size) * 0.02

    else:
        window_size = WINDOW_SIZE_SAMPLES_PREFIX
        energy_start_sec = lambda i: i * 0.02
        energy_end_sec = lambda i: i * 0.02 + 1

    transition_indices = np.array([i for i in range(0, number_of_tokens - window_size, HOP_SIZE_SAMPLES)])

    # remove the indices that will cause silent transition
    transition_indices = transition_indices[
        [np.mean(song.get_audio_energy_array(song.get_partial_audio(start_sec=energy_start_sec(i),
                                                                    end_sec=energy_end_sec(i)))[0]) > ENERGY_THRESHOLD
         for i in transition_indices]]

    return transition_indices


def connect_between_songs(song1: Song, song2: Song, model, use_accompaniment=False):
    assert song1.sr == song2.sr
    import math
    print("connect")

    print("encode audio to tokens")
    audio_encoder = model.audio_encoder
    text_tokens = [0]  # pad token id
    text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))

    if use_accompaniment:
        suffix = song1.suffix_accompaniment.audio
        prefix = song2.prefix_accompaniment.audio
    else:
        suffix = song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS)
        prefix = song2.get_partial_audio(end_sec=FULL_WINDOW_SECONDS)

    with torch.no_grad():
        prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
        suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))

    best_prob = -np.inf
    best_tuple = (suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, 0)

    print("create batches")

    suffix_transition_indices = get_transition_indices_list(song1, suffix_tokens.audio_codes.shape[-1], suffix=True)
    prefix_transition_indices = get_transition_indices_list(song2, prefix_tokens.audio_codes.shape[-1], suffix=False)

    number_of_tuples_to_check = len(suffix_transition_indices) * len(prefix_transition_indices)
    number_of_batches = math.ceil(number_of_tuples_to_check / BATCH_SIZE)

    batch_size_suffix = math.ceil(len(suffix_transition_indices) / number_of_batches)

    partial_prefix_tokens = np.array([prefix_tokens.audio_codes[..., i2:i2 + WINDOW_SIZE_SAMPLES_PREFIX][0, 0]
                                      for i2 in prefix_transition_indices])
    partial_prefix_tokens_batched = np.tile(partial_prefix_tokens, (batch_size_suffix, 1, 1))
    partial_prefix_tokens_batched = partial_prefix_tokens_batched.reshape(partial_prefix_tokens_batched.shape[0] *
                                                                          partial_prefix_tokens_batched.shape[1],
                                                                          partial_prefix_tokens_batched.shape[2])

    print(f"Number of batches: {number_of_batches}, number of pairs to check: {number_of_tuples_to_check}")

    print("start magic")
    for batch_number in range(number_of_batches):
        print(f"Batch #{batch_number}")
        suffix_transition_indices_batched = suffix_transition_indices[
                                                            batch_number * batch_size_suffix:
                                                            (batch_number + 1) * batch_size_suffix]
        if len(suffix_transition_indices_batched) == 0:
            break
        prefix_transition_indices_batched = prefix_transition_indices[:]

        partial_suffix_tokens = np.array([suffix_tokens.audio_codes[..., i1:i1+WINDOW_SIZE_SAMPLES_SUFFIX][0, 0]
                                          for i1 in suffix_transition_indices_batched])

        partial_suffix_tokens_batched = np.repeat(partial_suffix_tokens, partial_prefix_tokens.shape[0], axis=0)
        if partial_suffix_tokens.shape[0] != batch_size_suffix:  # can be at the last iteration
            partial_prefix_tokens_batched = np.tile(partial_prefix_tokens, (partial_suffix_tokens.shape[0], 1, 1))
            partial_prefix_tokens_batched = partial_prefix_tokens_batched.reshape(
                partial_prefix_tokens_batched.shape[0] *
                partial_prefix_tokens_batched.shape[1],
                partial_prefix_tokens_batched.shape[2])

        partial_suffix_tokens_batched = partial_suffix_tokens_batched.reshape(partial_suffix_tokens_batched.shape[0] *
                                                                              partial_suffix_tokens_batched.shape[1],
                                                                              partial_suffix_tokens_batched.shape[2])

        tuples_batched = np.array([np.repeat(suffix_transition_indices_batched, partial_prefix_tokens.shape[0]),
                                   np.tile(prefix_transition_indices_batched, partial_suffix_tokens.shape[0])]).T

        log_sum = calculate_log_prob_of_sequence_given_another_sequence(torch.from_numpy(partial_suffix_tokens_batched),
                                                                        torch.from_numpy(partial_prefix_tokens_batched),
                                                                        model, text_tokens)
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


def create_full_playlist(songs_dir, outpath='/home/joberant/NLP_2324/yaelshemesh/outputs/haviv_10/musicgen_spleeter/',  use_accompaniment=False):
    print("create_play list")
    print("load_model")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(number_of_songs)]
    
    if use_accompaniment:
        from utils import songs_spleeter
        songs_spleeter(songs_list, time_in_sec=FULL_WINDOW_SECONDS, spleeter_output_dir_path='/home/joberant/NLP_2324/yaelshemesh/spleeter_output')

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
            best_prob, best_tuple = connect_between_songs(song1, song2, model, use_accompaniment=use_accompaniment)
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

    try:
        np.save(os.path.join(outpath, f'playlister_playlist_numpy.npy'), full_playlist_audio)
        np.save(os.path.join(outpath, f'playlister_playlist_fader_numpy.npy'), full_playlist_audio_fader)
        np.save(os.path.join(outpath, f'songs_name_order.npy'), np.array(songs_name_order))
        np.save(os.path.join(outpath, f'adjacency_matrix.npy'), adjacency_matrix)
        np.save(os.path.join(outpath, f'cut_indices_suffix_window_{WINDOW_SIZE_SAMPLES_SUFFIX}_hop_{HOP_SIZE_SAMPLES}.npy'),
                cut_indices_suffix)
        np.save(os.path.join(outpath, f'cut_indices_prefix_window_{WINDOW_SIZE_SAMPLES_PREFIX}_hop_{HOP_SIZE_SAMPLES}.npy'),
                cut_indices_prefix)

    except Exception as e:
        print("Failed to save the numpy arrays")

    save_audio_file(os.path.join(outpath, f'playlister_playlist.wav'), full_playlist_audio, songs_list[0].sr)
    save_audio_file(os.path.join(outpath, f'playlister_playlist_fader.wav'), full_playlist_audio_fader, songs_list[0].sr)


if __name__ == '__main__':
    create_full_playlist('/home/joberant/NLP_2324/yaelshemesh/haviv_3',outpath='/home/joberant/NLP_2324/yaelshemesh/playlyster_outputs',  use_accompaniment=True)
    # song1 = Song("eyal\\yafyufa.mp3", sr=32000)
    # song2 = Song("eyal\\malkat hayofi.mp3", sr=32000)
    #
    # print("load_model")
    # model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    #
    # connect_between_songs(song1, song2, model)
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



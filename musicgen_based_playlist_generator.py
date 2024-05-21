import os
import numpy as np
from song_handler import Song
import torch
from utils import Graph, fadeout_cur_fadein_next, save_audio_file
from transformers import MusicgenForConditionalGeneration

PATH = '/home/joberant/NLP_2324/yaelshemesh'
DEFAULT_OUT_PATH = 'outputs'
DEFAULT_SONGS_DIR = 'yael_playlist'

DEVICE = 'cuda'

FADE_DURATION = 2.0
NUMBER_OF_CODEBOOKS = 4

HOP_SIZE_SAMPLES = 25  # 0.5 sec / 0.02
WINDOW_SIZE_SAMPLES_SUFFIX = 200  # 4 sec/ 0.02
WINDOW_SIZE_SAMPLES_PREFIX = 100  # 2 sec/ 0.02

FULL_WINDOW_SECONDS = 45

BATCH_SIZE = 25
ENERGY_THRESHOLD = 10

TOKEN_LENGTH_IN_SECONDS = 0.02


def calculate_log_prob_of_sequence_given_another_sequence(token_sequence_1, token_sequence_2, model, text_tokens=None,
                                                          codebook=0):
    if text_tokens is None:
        text_tokens = [0]  # pad token id
        text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))

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

    # use only the first token
    batch_sequence_2_logmax = sequence_2_logmax[range(codebook, logits.shape[0], NUMBER_OF_CODEBOOKS)]

    return torch.sum(batch_sequence_2_logmax, dim=-1)


def calculate_log_prob_of_sequence_given_another_sequence_method_2(token_sequence_1, token_sequence_2, model,
                                                                   text_tokens=None):
    if text_tokens is None:
        text_tokens = [0]  # pad token id
        text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))
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


def connect_between_songs(song1: Song, song2: Song, home_directory, use_accompaniment=False):
    assert song1.sr == song2.sr
    import math
    print("connect")
    print("load_model")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", cache_dir=home_directory)
    print("encode")
    audio_encoder = model.audio_encoder

    if use_accompaniment:
        suffix = song1.suffix_accompaniment.audio
        prefix = song2.prefix_accompaniment.audio
    else:
        suffix = song1.get_partial_audio(start_sec=-FULL_WINDOW_SECONDS)
        prefix = song2.get_partial_audio(end_sec=FULL_WINDOW_SECONDS)

    prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
    suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))

    best_prob = -np.inf
    best_tuple = (suffix_tokens.audio_codes.shape[-1] - WINDOW_SIZE_SAMPLES_SUFFIX, 0)

    suffix_transition_indices = get_transition_indices_list(song1, suffix_tokens.audio_codes.shape[-1], suffix=True)
    prefix_transition_indices = get_transition_indices_list(song2, prefix_tokens.audio_codes.shape[-1], suffix=False)

    number_of_tuples_to_check = len(suffix_transition_indices) * len(prefix_transition_indices)
    number_of_batches = math.ceil(number_of_tuples_to_check / BATCH_SIZE)

    partial_suffix_tokens_all_batches = torch.tensor([], dtype=torch.int)
    partial_prefix_tokens_all_batches = torch.tensor([], dtype=torch.int)
    tuples = []
    print("create batches")
    for idx, i1 in enumerate(suffix_transition_indices):
        print(f'{idx} / {len(suffix_transition_indices)}')
        for i2 in prefix_transition_indices:
            tuples.append((i1, i2))
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i1:i1+WINDOW_SIZE_SAMPLES_SUFFIX][0, 0]
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i2:i2+WINDOW_SIZE_SAMPLES_PREFIX][0, 0]

            partial_suffix_tokens_all_batches = torch.cat([partial_suffix_tokens_all_batches, partial_suffix_tokens], dim=0)
            partial_prefix_tokens_all_batches = torch.cat([partial_prefix_tokens_all_batches, partial_prefix_tokens], dim=0)

    print(f"Number of batches: {number_of_batches}, number of pairs to check: {number_of_tuples_to_check}")
    print("start magic")
    for batch_number in range(math.ceil(partial_prefix_tokens_all_batches.shape[0]/ (BATCH_SIZE * 4))):
        print(f"Batch #{batch_number}")
        partial_suffix_tokens_batched = partial_suffix_tokens_all_batches[batch_number * BATCH_SIZE * 4: (batch_number + 1) * BATCH_SIZE * 4]
        partial_prefix_tokens_batched = partial_prefix_tokens_all_batches[batch_number * BATCH_SIZE * 4: (batch_number + 1) * BATCH_SIZE * 4]
        tuples_batched = tuples[batch_number * BATCH_SIZE: (batch_number + 1) * BATCH_SIZE]
        log_sum = calculate_log_prob_of_sequence_given_another_sequence(partial_suffix_tokens_batched,
                                                                        partial_prefix_tokens_batched, model)
        cur_best_prob = torch.max(log_sum)
        cur_best_tuple = tuples_batched[torch.argmax(log_sum)]
        if cur_best_prob > best_prob:
            best_prob = cur_best_prob
            best_tuple = cur_best_tuple

    print(f"Total Log Sum Probability: {best_prob}")
    print(f"Best tuple: {best_tuple}")

    return best_prob, best_tuple


def create_full_playlist(songs_dir, outpath, home_directory=PATH, use_accompaniment=False, fade_duration=FADE_DURATION):
    print("create_play list")
    number_of_songs = len(os.listdir(songs_dir))
    file_names_list = os.listdir(songs_dir)
    songs_list = [Song(os.path.join(songs_dir, file_names_list[i]), sr=32000) for i in range(number_of_songs)]

    if use_accompaniment:
        from utils import songs_spleeter
        spleeter_output_dir_path = os.path.join(home_directory, 'spleeter_output')
        os.makedirs(spleeter_output_dir_path, exist_ok=True)
        songs_spleeter(songs_list, time_in_sec=FULL_WINDOW_SECONDS, spleeter_output_dir_path=spleeter_output_dir_path)

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
            best_prob, best_tuple = connect_between_songs(song1, song2, home_directory,
                                                          use_accompaniment=use_accompaniment)
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
                                                                32000, duration=fade_duration)
        else:
            full_playlist_audio_fader = curr_song_partial_audio

    try:
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
    import argparse
    parser = argparse.ArgumentParser(description='Playlister MusicGen implementation. Saves the playlist as one audio file.')
    parser.add_argument("--home_dir", type=str, default=PATH, help='model cache will be saved here')
    parser.add_argument("--songs_dir", type=str, default=DEFAULT_SONGS_DIR, help='directory with songs that you wish to create a playlist out of them')
    parser.add_argument("--outpath", type=str, default=DEFAULT_OUT_PATH, help='the output playlist and numpy arrays will be saved here')
    parser.add_argument("--fade_duration", type=float, default=FADE_DURATION, help='fade duration. used in playlister_playlist_fader.wav file')
    parser.add_argument("--use_spleeter", action='store_true', default=False,
                        help='whether to use spleeter and choose the transition point based only on the accompaniment or not')

    args = parser.parse_args()
    os.environ['HF_HOME'] = args.home_dir

    create_full_playlist(args.songs_dir, args.outpath, home_directory=args.home_dir,
                         use_accompaniment=args.use_spleeter, fade_duration=args.fade_duration)

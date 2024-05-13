import os

import numpy as np
from transformers import JukeboxModel, JukeboxConfig, JukeboxPriorConfig, JukeboxPrior, JukeboxVQVAE
from song_handler import Song
import soundfile as sf
import torch

FADE_DURATION = 3.0


def calculate_log_prob_of_sequence_given_another_sequence(token_sequence_1, token_sequence_2, model, text_tokens):
    tokens = torch.cat([token_sequence_1, token_sequence_2], dim=-1)

    with torch.no_grad():
        outputs = model(input_ids=text_tokens, decoder_input_ids=tokens)
        logits = outputs.logits

    log_sum = 0
    range_index = range(max(token_sequence_1.shape[1] - 1, 0), tokens.shape[1] - 1)
    for i in range_index:
        past_tok, current_tok = i, i + 1
        log_token_prob = 0
        for token_dim_index in range(1):
            token_logit = logits[token_dim_index, past_tok, :]
            token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
            log_token_prob += token_log_probs[tokens[token_dim_index, current_tok]].item()

        log_sum += log_token_prob
        # print(f"Token, Log Prob: {log_token_prob}")
    return log_sum


def calculate_log_prob_of_sequence_given_another_sequence_method_2(token_sequence_1, token_sequence_2, model):
    pass


def connect_between_songs_first_try(song1: Song, song2: Song):
    assert song1.sr == song2.sr
    from transformers import AutoProcessor, MusicgenModel, AutoTokenizer, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    audio_encoder = model.audio_encoder
    text_tokens = tokenizer.encode("")
    text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))

    prefix = song2.get_partial_audio(end_sec=30)
    suffix = song1.get_partial_audio(start_sec=-30)

    prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
    suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))

    hop_size_samples = 50  # 1 sec / 0.02
    window_size_samples_suffix = 200  # 4 sec/ 0.02
    window_size_samples_prefix = 100  # 2 sec/ 0.02

    best_prob = -np.inf
    best_tuple = None
    for i1 in range(0, suffix_tokens.audio_codes.shape[-1] - window_size_samples_suffix, hop_size_samples):
        for i2 in range(0, prefix_tokens.audio_codes.shape[-1] - window_size_samples_prefix, hop_size_samples):
            print(f'{i1, i2} / {suffix_tokens.audio_codes.shape[-1] - window_size_samples_suffix, prefix_tokens.audio_codes.shape[-1] - window_size_samples_prefix}')
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i1:i1+window_size_samples_suffix][0, 0]
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i2:i2+window_size_samples_prefix][0, 0]
            log_sum = calculate_log_prob_of_sequence_given_another_sequence(partial_suffix_tokens, partial_prefix_tokens, model, text_tokens)
            print(f"Total Log Sum Probability: {log_sum}")

            if log_sum > best_prob:
                best_prob = log_sum
                best_tuple = (i1, i2)
                print("FOUND NEW BEST TUPLE !")
                concat_audio = np.concatenate(
                    [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - 30 + (best_tuple[0] + window_size_samples_suffix) * 0.02),
                     song2.get_partial_audio(start_sec=best_tuple[1] * 0.02)])
                sf.write(f'{song1.song_name} + {song2.song_name}_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)

                concat_audio = np.concatenate(
                    [song1.get_partial_audio(start_sec=len(song1.audio) / song1.sr - 30 + best_tuple[0] * 0.02,
                                             end_sec=len(song1.audio) / song1.sr - 30 + (best_tuple[0] + window_size_samples_suffix) * 0.02),
                     song2.get_partial_audio(start_sec=best_tuple[1] * 0.02,
                                             end_sec=best_tuple[1] * 0.02 + window_size_samples_prefix * 0.02)])
                sf.write(f'{song1.song_name} + {song2.song_name}_partial_{best_tuple}_{best_prob}.wav', concat_audio, song1.sr)

        print(f'Best indices: {best_tuple}')
    # concat_audio = np.concatenate(
    #     [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - 30 + best_tuple[0]*0.02),
    #      song2.get_partial_audio(start_sec=best_tuple[1]*0.02)])

    concat_audio = fadeout_cur_fadein_next(
        song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - 30 + (best_tuple[0] + window_size_samples_suffix) * 0.02 + FADE_DURATION /2, len(song1.audio))),
        song2.get_partial_audio(start_sec=max(best_tuple[1]*0.02 - FADE_DURATION /2, 0)), song1.sr,
        duration=FADE_DURATION)

    sf.write(f'{song1.song_name} + {song2.song_name}_long_fader_more_audio.wav', concat_audio, song1.sr)

    concat_audio = fadeout_cur_fadein_next(
        song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - 30 + (best_tuple[0] + window_size_samples_suffix) * 0.02, len(song1.audio))),
        song2.get_partial_audio(start_sec=best_tuple[1]*0.02), song1.sr,
        duration=FADE_DURATION)

    sf.write(f'{song1.song_name} + {song2.song_name}_long_fader.wav', concat_audio, song1.sr)

    concat_audio = fadeout_cur_fadein_next(
        song1.get_partial_audio(end_sec=min(len(song1.audio) / song1.sr - 30 + (best_tuple[0] + window_size_samples_suffix) * 0.02 + 2 /2, len(song1.audio))),
        song2.get_partial_audio(start_sec=max(best_tuple[1]*0.02 - 2 /2, 0)), song1.sr,
        duration=2)

    sf.write(f'{song1.song_name} + {song2.song_name}_short_fader_more_audio.wav', concat_audio, song1.sr)

    concat_audio = np.concatenate(
        [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - 30 + (
                                             best_tuple[0] + window_size_samples_suffix) * 0.02),
         song2.get_partial_audio(start_sec=best_tuple[1] * 0.02)])

    sf.write(f'{song1.song_name} + {song2.song_name}_no fader.wav', concat_audio, song1.sr)


def connect_between_songs_second_try(song1: Song, song2: Song):
    assert song1.sr == song2.sr
    from transformers import AutoProcessor, MusicgenModel, AutoTokenizer, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    audio_encoder = model.audio_encoder

    prefix = song2.get_partial_audio(end_sec=30)
    suffix = song1.get_partial_audio(start_sec=-30)

    prefix_tokens = audio_encoder.encode(torch.from_numpy(prefix.reshape(1, 1, len(prefix))))
    suffix_tokens = audio_encoder.encode(torch.from_numpy(suffix.reshape(1, 1, len(suffix))))
    text_tokens = tokenizer.encode("continue the song")
    text_tokens = torch.tensor(text_tokens).reshape((1, len(text_tokens)))
    hop_size_samples = 25  # 0.5 sec / 0.02
    window_size_samples = 100  # 2 sec/ 0.02

    best_prob = -np.inf
    best_tuple = None
    for i1 in range(0, suffix_tokens.audio_codes.shape[-1] - window_size_samples, hop_size_samples):
        for i2 in range(0, prefix_tokens.audio_codes.shape[-1] - window_size_samples, hop_size_samples):
            print(f'{i1, i2} / {suffix_tokens.audio_codes.shape[-1] - window_size_samples, prefix_tokens.audio_codes.shape[-1] - window_size_samples}')
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i1:i1+window_size_samples][0, 0]
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i2:i2 + window_size_samples][0, 0]
            tokens = torch.cat([partial_suffix_tokens, partial_prefix_tokens], dim=-1)

            log_sum = 0
            # loop for every token in prefix
            for i in range(0, window_size_samples):
                curr_tokens = tokens[..., 0: window_size_samples + i]
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

            print(f"Total Log Sum Probability: {log_sum}")

            if log_sum > best_prob:
                best_prob = log_sum
                best_tuple = (i1, i2)

        print(f'Best indices: {best_tuple}')
    concat_audio = np.concatenate(
        [song1.get_partial_audio(end_sec=len(song1.audio) / song1.sr - 30 + best_tuple[0]*0.02),
         song2.get_partial_audio(start_sec=best_tuple[1]*0.02)])

    sf.write(f'{song1.song_name} + {song2.song_name}.wav', concat_audio, song1.sr)


def fadeout_cur_fadein_next(audio1, audio2, sr, duration=FADE_DURATION):
    apply_fadeout(audio1, sr, duration)
    apply_fadein(audio2, sr, duration)
    length = int(duration*sr)
    new_audio = audio1
    end = new_audio.shape[0]
    start = end - length
    new_audio[start:end] += audio2[:length]
    new_audio = np.concatenate((new_audio, audio2[length:]))
    return new_audio


def apply_fadeout(audio, sr, duration=FADE_DURATION):
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def apply_fadein(audio, sr, duration=FADE_DURATION):
    length = int(duration*sr)
    # linear fade
    fade_curve = np.linspace(0.0, 1.0, length)
    # apply the curve
    audio[:length] = audio[:length] * fade_curve


if __name__ == '__main__':
    songs_pairs = np.array([[(song_name_1, song_name_2) for song_name_1 in os.listdir("../eyal")] for song_name_2 in os.listdir("../eyal")])
    songs_pairs = songs_pairs.reshape((songs_pairs.shape[0] * songs_pairs.shape[1], 2))
    np.random.shuffle(songs_pairs)

    for song_name_1, song_name_2 in songs_pairs:
        print(song_name_1, song_name_2)
        if song_name_1 == song_name_2:
            continue
        try:
            song1 = Song(f"../eyal/{song_name_1}", sr=32000)
            song2 = Song(f"../eyal/{song_name_2}", sr=32000)

            # song1 = Song("../songs/Wish You Were Here - Incubus - Lyrics.mp3", sr=32000)
            # song2 = Song("../songs/Incubus - Drive.mp3", sr=32000)

            connect_between_songs_first_try(song1, song2)
        except Exception as e:
            continue


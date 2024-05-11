import numpy as np
from transformers import JukeboxModel, JukeboxConfig, JukeboxPriorConfig, JukeboxPrior, JukeboxVQVAE
from song_handler import Song
import soundfile as sf
import torch


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

    hop_size_samples = 50  # 1 sec / 0.02
    window_size_samples = 100  # 2 sec/ 0.02

    best_prob = -np.inf
    best_tuple = None
    for i1 in range(0, suffix_tokens.audio_codes.shape[-1] - window_size_samples, hop_size_samples):
        for i2 in range(0, prefix_tokens.audio_codes.shape[-1] - window_size_samples, hop_size_samples):
            print(f'{i1, i2} / {suffix_tokens.audio_codes.shape[-1] - window_size_samples, prefix_tokens.audio_codes.shape[-1] - window_size_samples}')
            partial_prefix_tokens = prefix_tokens.audio_codes[..., i1:i1+window_size_samples][0, 0]
            partial_suffix_tokens = suffix_tokens.audio_codes[..., i2:i2+window_size_samples][0, 0]
            tokens = torch.cat([partial_suffix_tokens, partial_prefix_tokens], dim=-1)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([[0]]), decoder_input_ids=tokens)
                logits = outputs.logits

            log_sum = 0
            range_index = range(partial_suffix_tokens.shape[1] - 1, tokens.shape[1] - 1)
            for i in range_index:
                past_tok, current_tok = i, i + 1
                log_token_prob = 0
                for token_dim_index in range(1):
                    token_logit = logits[token_dim_index, past_tok, :]
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

    sf.write('music_gen_audioo.wav', concat_audio, song1.sr)


if __name__ == '__main__':
    # song1 = Song("../songs/lev_shel_gever.mp3", sr=32000)
    # song2 = Song("../songs/biladaih.mp3", sr=32000)

    song1 = Song("../songs/Wish You Were Here - Incubus - Lyrics.mp3", sr=32000)
    song2 = Song("../songs/Incubus - Drive.mp3", sr=32000)

    connect_between_songs_second_try(song1, song2)


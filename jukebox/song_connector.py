from transformers import JukeboxModel, JukeboxConfig, JukeboxPriorConfig, JukeboxPrior, JukeboxVQVAE
from song_handler import Song
import soundfile as sf
import torch
import os

PATH = '/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

DEVICE = 'cuda:2'

def connect_between_songs(song1: Song, song2: Song):
    # Use a pipeline as a high-level helper
    from transformers import pipeline
    model = JukeboxPrior.from_pretrained("openai/jukebox-1b-lyrics", cache_dir=PATH, min_duration=0, config=None).eval()
    song_1_audio = torch.from_numpy(song1.get_partial_audio(start_sec=-30, end_sec=-26))
    song_1_audio_next = torch.from_numpy(song1.get_partial_audio(start_sec=-26, end_sec=-24))
    song_2_audio = torch.from_numpy(song2.get_partial_audio(start_sec=28, end_sec=30))
    all_audio_option1 = torch.cat([song_1_audio, song_2_audio])
    all_audio_option2 = torch.cat([song_1_audio, song_1_audio_next])
    if torch.cuda.is_available():
        all_audio_option1.to(DEVICE)
        all_audio_option2.to(DEVICE)
        model.to(DEVICE)

    out = model(hidden_states=all_audio_option1)

    print(1)


def connect_between_songs_music_lm(song: Song):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", return_dict_in_generate=True)
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", return_dict_in_generate=True)

    inputs = processor(
        audio=song.audio,
        sampling_rate=song.sr,
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, do_sample=True, output_scores=True)
    sf.write('music_gen.wav', audio_values[0, 0].numpy(), model.config.audio_encoder.sampling_rate)


if __name__ == '__main__':
    song1 = Song("/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/haviv_3/KONGOS - Come with Me Now.mp3", sr=44100)
    song2 = Song("/vol/joberant_nobck/data/NLP_368307701_2324/yaelshemesh/haviv_3/Jay-Z, Linkin Park - Numb _ Encore my-free-mp3s.com .mp3", sr=44100)

    connect_between_songs(song1, song2)


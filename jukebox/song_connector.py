from transformers import JukeboxModel, JukeboxConfig, JukeboxPriorConfig, JukeboxPrior, JukeboxVQVAE
from song_handler import Song
import soundfile as sf
import torch


def connect_between_songs(song1: Song, song2: Song):
    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("feature-extraction", model="openai/jukebox-1b-lyrics")

    assert song1.sr == song2.sr, "sample rate should be the same in both songs"
    vqvae_model = JukeboxVQVAE.from_pretrained("openai/jukebox-5b").eval()

    config = JukeboxPriorConfig(sampling_rate=song1.sr, metadata_conditioning=False)

    model_level_0 = JukeboxPrior(config, vqvae_decoder=vqvae_model.decode, vqvae_encoder=vqvae_model.encode)
    next_token = model_level_0.forward(hidden_states=torch.from_numpy(song1.audio), metadata=None, get_preds=True)
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
    song1 = Song("../songs/not afraid.mp3")
    song2 = Song("../songs/biladaih.mp3")
    end_of_song1_audio = song1.get_partial_audio(-6,-2)
    sf.write('not afraid.wav', end_of_song1_audio, song1.sr)

    end_of_song1 = Song('not afraid.wav', sr=32000)
    connect_between_songs_music_lm(end_of_song1)


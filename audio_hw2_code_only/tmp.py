
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import soundfile as sf
from dtw import *
import webrtcvad
import contextlib
import wave

music_path = r'C:\Users\ofirn\Music\songs'



def get_mel_spec_dict(audio_signals, sr=16000, win_length_sec=0.025, hop_length_sec=0.01, n_filters=80):
    window_length_samples = int(win_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)
    n_fft = window_length_samples

    mel_spec_list = []
    for audio in audio_signals:
        mel_signal = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                    win_length=window_length_samples, hop_length=hop_length_samples,
                                                    n_fft=n_fft, n_mels=n_filters)
        mel_spec_list.append(mel_signal)
    return mel_spec_list

# Get all files in the music_path directory
files = os.listdir(music_path)
prefixes = []

# Iterate over each file
for i in range(len(files[0:4])):
    size = 10
    # Load the audio file using librosa
    file = files[i]
    print(file)
    audio, sr = librosa.load(os.path.join(music_path, file))

    # Extract the first 10 seconds of the audio
    first_10_seconds = audio[sr*5:sr * (5+size)]

    # # Extract the last 10 seconds of the audio
    # last_10_seconds = audio[-sr * size:]

    # Play the first 10 seconds of the audio
    sf.write(f'first_{size}_seconds_{i}_{file.split(".")[0]}.wav', first_10_seconds, sr)
    # os.system('start first_10_seconds.wav')
    prefixes.append(f'first_{size}_seconds_{i}_{file.split(".")[0]}.wav')


# go over every pair of prefixes and compare them using dtw
k=0
for i in range(len(prefixes)):
    for j in range(i+1, len(prefixes)):
        # Load the audio files
        audio1, sr1 = librosa.load(prefixes[i])
        audio2, sr2 = librosa.load(prefixes[j])
        mel_specs = get_mel_spec_dict([audio1, audio2])

        # Create a DTW object
        # dtw = DTW()
        # dtw_distance = dtw.calculate_dtw_distance(mel_specs[0], mel_specs[1])
        dtw_matrix, wp = librosa.sequence.dtw(mel_specs[0], mel_specs[1], subseq=True)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(dtw_matrix, x_axis='frames', y_axis='frames', ax=ax)
        ax.set(title='DTW cost', xlabel=f'{prefixes[i]}', ylabel=f'{prefixes[j]}')
        ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
        ax.legend()
        fig.colorbar(img, ax=ax)
        plt.savefig(f'dtw_{k}.png')
        k+=1
        print(dtw_matrix)




def convert_mp3_to_wav(mp3_path, wav_path):
    audio, sr = librosa.load(mp3_path, sr=48000)
    sf.write(wav_path, audio, sr)


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate
    

def check_vocals_intervals_in_song(path):
    audio, sr =read_wave(path)
    vad = webrtcvad.Vad(3)   
    frame_duration = 10  # ms
    frame_size = int(sr * (frame_duration / 1000)*2)
    frames = [audio[i: i + frame_size] for i in range(0, len(audio), frame_size)]
    vocals_intervals = []
    for i, frame in enumerate(frames):
        if (len(frame) < frame_size):
            # add silence to the end of the frame
            bytes_to_add = b'\x00' * (frame_size - len(frame))
            frame += bytes_to_add
        frame = frame
        if not vad.is_speech(frame, sr):
            if not vocals_intervals or vocals_intervals[-1][1] != i - 1:
                vocals_intervals.append([i, i])
            else:
                vocals_intervals[-1][1] = i
    # convert frames to seconds
    vocals_intervals = [[interval[0] * frame_duration / 1000, interval[1] * frame_duration / 1000] for interval in vocals_intervals]
    return vocals_intervals


print(k)
        




#TODO
# 1. adjust dtw to all kind of errors
# 2. find the alignment along the way using dtw algorithm and extract the time of the alignment

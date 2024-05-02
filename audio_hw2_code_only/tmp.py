
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import soundfile as sf
from dtw import *

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

print(k)
        


#TODO
# 1. adjust dtw to all kind of errors
# 2. find the alignment along the way using dtw algorithm and extract the time of the alignment

# import librosa
# import scipy
# import matplotlib.pyplot as plt
# import numpy as np
# import parselmouth
# import soundfile as sf
#
#
# def draw_pitch(audio, ax, sr):
#     sound = parselmouth.Sound(audio, sampling_frequency=sr)
#     pitch = sound.to_pitch()
#     pitch_values = pitch.selected_array['frequency']
#     pitch_values[pitch_values == 0] = np.nan
#     ax.plot(pitch.xs(), pitch_values + 500, 'o', markersize=1.5, color='w')
#     ax.plot(pitch.xs(), pitch_values + 500, 'o', markersize=0.75)
#
#
# def load_audio_file(audio_file_path: str):
#     audio, sr = librosa.load(audio_file_path, sr=48000)
#     print(f"Sampling rate is: {sr}")
#     return audio
#
#
# def resample_audio(original_sr, output_sr, audio):
#     secs = len(audio) / original_sr  # number of seconds in the audio
#     samples = int(secs * output_sr)  # number of samples to resample
#     return np.array(scipy.signal.resample(audio, samples), dtype=np.float32)
#     # return librosa.resample(audio, orig_sr=original_sr, target_sr=output_sr)
#
#
# def plot(audio, sr, win_length_sec=0.02, hop_length_sec=0.01):  # question 1.d
#     fig, axs = plt.subplots(2, 2)#, sharex='all')
#
#     # audio plot
#     seconds = np.array(range(len(audio))) / sr
#     axs[0, 0].plot(seconds, audio)
#     axs[0, 0].set_title('Audio')
#     axs[0, 0].set_xlabel('Time [secs]')
#     axs[0, 0].set_ylabel('Amplitude')
#
#     # spectogram
#     window_length_samples = int(win_length_sec * sr)
#     hop_length_samples = int(hop_length_sec * sr)
#     n_fft = window_length_samples
#     stft = librosa.stft(audio, n_fft=n_fft, win_length=window_length_samples, hop_length=hop_length_samples)
#     stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
#     img = librosa.display.specshow(stft_db, sr=sr, y_axis='linear', x_axis='time', ax=axs[0, 1],
#                                    hop_length=hop_length_samples)
#     fig.colorbar(img, format='%+2.0f dB', ax=axs[0, 1])
#     axs[0, 1].set_xlabel('Time [secs]')
#     axs[0, 1].set_ylabel('Frequency [Hz]')
#     axs[0, 1].set_title('Spectogram')
#     draw_pitch(audio, axs[0, 1], sr)
#
#     # mel-spectogram
#     mel_signal = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=window_length_samples,
#                                                 hop_length=hop_length_samples, n_fft=n_fft, n_mels=80)
#
#     # mel_signal = np.abs(stft)**2
#     # mel_signal = librosa.feature.melspectrogram(S=mel_signal, sr=sr)
#     power_to_db = librosa.power_to_db(np.abs(mel_signal), ref=np.max)
#     img = librosa.display.specshow(power_to_db, sr=sr, y_axis='mel', x_axis='time', ax=axs[1, 1],
#                                    hop_length=hop_length_samples)
#
#     # sgram_mag, _ = librosa.magphase(stft)
#     # mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
#     # mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
#     # librosa.display.specshow(mel_sgram, y_axis='mel', x_axis='time', ax=axs[1, 1],
#     #                                hop_length=hop_length_samples)
#
#     fig.colorbar(img, format='%+2.0f dB', ax=axs[1, 1])
#
#     axs[1, 1].set_xlabel('Time [secs]')
#     axs[1, 1].set_ylabel('Frequency [Hz]')
#     axs[1, 1].set_title('Mel-Spectogram')
#
#     # energy and RMS
#
#     energy = np.array([
#         sum(abs(audio[i:i + window_length_samples] ** 2))
#         for i in range(0, len(audio), hop_length_samples)
#     ])
#     energy = 10 * np.log10(energy[:-3])
#     t = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length_samples)
#
#     #rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length, center=True)
#
#     rms = (np.sum(audio ** 2) / len(audio)) ** 0.5
#     rms = 20 * np.log10(rms)
#
#     axs[1, 0].plot(t, energy, label='energy')
#     axs[1, 0].plot([0, seconds[-1]], [rms, rms], label='rms')
#     axs[1, 0].set_title('Energy and RMS')
#     axs[1, 0].set_xlabel('Time [secs]')
#     axs[1, 0].set_ylabel('Energy [dB]')
#     axs[1, 0].legend()
#     plt.show()
#
#
# if __name__ == '__main__':
#     # ------------------------
#     # question 1
#     # ------------------------
#
#     # a ----------------------
#     # ii. Sampling rate is 48000 (I exported the wav file from Audacity)
#     original_audio = load_audio_file('recording.wav')
#
#     # b ----------------------
#     resampled_audio_32 = resample_audio(48000, 32000, original_audio)
#
#     # c ----------------------
#     resampled_audio_16_1 = np.array([resampled_audio_32[i] for i in range(len(resampled_audio_32)) if i % 2 == 0],
#                                     dtype=np.float32)
#     resampled_audio_16_2 = resample_audio(32000, 16000, resampled_audio_32)
#
#     # ------------------------
#     # question 1.c experiment
#     # ------------------------
#     #
#     # resampled_audio_8_1 = np.array([resampled_audio_32[i] for i in range(len(resampled_audio_32)) if i % 4 == 0],
#     #                                 dtype=np.float32)
#     # resampled_audio_8_2 = resample_audio(48000, 8000, original_audio)
#     #
#     # plot(resampled_audio_8_1, 8000)
#     # sf.write('resampled_audio_8_1.wav', resampled_audio_8_1, 8000)
#     # plot(resampled_audio_8_2, 8000)
#     # sf.write('resampled_audio_8_2.wav', resampled_audio_8_2, 8000)
#
#     # e ----------------------
#     # plot(resampled_audio_32, 32000)
#     # plot(resampled_audio_16_1, 16000)
#     sf.write('resampled_audio_16_1.wav', resampled_audio_16_1, 16000)
#     # plot(resampled_audio_16_2, 16000)
#     sf.write('resampled_audio_16_2.wav', resampled_audio_16_2, 16000)
#
#     # ------------------------
#     # question 2
#     # ------------------------
#
#     # a ----------------------
#     noise_original_audio = load_audio_file('stationary_noise.wav')
#     noise_resampled_audio_16 = resample_audio(48000, 16000, noise_original_audio)
#
#     # b ----------------------
#     noise_and_signal_16_1 = resampled_audio_16_1 + noise_resampled_audio_16[:len(resampled_audio_16_1)]
#     noise_and_signal_16_2 = resampled_audio_16_2 + noise_resampled_audio_16[:len(resampled_audio_16_2)]
#
#     # c ----------------------
#     plot(resampled_audio_16_2, 16000)
#     plot(noise_resampled_audio_16, 16000)
#     plot(noise_and_signal_16_2, 16000)
#
#     # ------------------------
#     # question 3
#     # ------------------------
#
#     # a ----------------------
#
#     print("Done")

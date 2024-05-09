import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
from audio_hw2_code_only.plotter import Plotter
import math
import copy


def load_audio_file(audio_file_path: str):
    audio, sr = librosa.load(audio_file_path, sr=48000)
    print(f"Sampling rate is: {sr}")
    return audio


def resample_audio(original_sr, output_sr, audio):
    secs = len(audio) / original_sr  # number of seconds in the audio
    samples = int(secs * output_sr)  # number of samples to resample
    return np.array(scipy.signal.resample(audio, samples), dtype=np.float32)


def spectral_subtraction(audio, threshold, sr=16000, win_length_sec=0.02, hop_length_sec=0.01, buffer_size_seconds=0.5):
    window_length_samples = int(win_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)

    n_fft = window_length_samples
    number_of_frames = math.ceil((len(audio) - window_length_samples) / hop_length_samples)

    stft = librosa.stft(audio, n_fft=n_fft, win_length=window_length_samples,
                        hop_length=hop_length_samples)[:, :number_of_frames]

    energy = np.array([sum(abs(audio[i:i + window_length_samples] ** 2))
                          for i in range(0, len(audio) - window_length_samples, hop_length_samples)])
    noise_map = (10 * np.log10(energy)) < threshold

    buffer_size_samples = int(buffer_size_seconds * sr) // hop_length_samples
    noise_map_cumsum = np.cumsum(noise_map)
    abs_stft = np.abs(stft)
    noise_footprints = np.array(
        [np.mean(abs_stft[:, noise_map][:, max(noise_map_cumsum[i] - buffer_size_samples, 0): noise_map_cumsum[i]], axis=1) for i in
         range(stft.shape[1])]).T

    new_stft = (abs_stft - noise_footprints) * np.exp(1j * np.angle(stft))
    new_audio = librosa.istft(new_stft, n_fft=n_fft, win_length=window_length_samples, hop_length=hop_length_samples)
    noise_audio = librosa.istft(noise_footprints, n_fft=n_fft, win_length=window_length_samples, hop_length=hop_length_samples)
    return new_audio, noise_audio


def auto_gain_control(audio, threshold, target_rms,  sr=16000, win_length_sec=0.02, hop_length_sec=0.01, buffer_size_seconds=1):
    window_length_samples = int(win_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)
    buffer_size_samples = int(buffer_size_seconds * sr)
    new_audio = copy.deepcopy(audio)

    scaling_factors = []

    for i in range(0, len(audio) - window_length_samples, hop_length_samples):
        frame_energy = sum(abs(audio[i:i + window_length_samples] ** 2))
        if (10 * np.log10(frame_energy)) < threshold:
            gain = 1
        else:
            buffer_rms = (np.sum(audio[max(i + window_length_samples-buffer_size_samples, 0):
                                       i + window_length_samples] ** 2) / ((i + window_length_samples) -
                          max(i + window_length_samples-buffer_size_samples, 0))) ** 0.5
            gain = target_rms / buffer_rms

        scaling_factors.append(gain)
        new_audio[i:i + window_length_samples] = audio[i:i + window_length_samples] * gain

    return new_audio, scaling_factors


def time_stretching_algorithm(audio, papping_function, sr=16000, win_length_sec=0.02, hop_length_sec=0.01):
    window_length_samples = int(win_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)

    n_fft = window_length_samples
    number_of_frames = math.ceil((len(audio) - window_length_samples) / hop_length_samples)

    stft = librosa.stft(audio, n_fft=n_fft, win_length=window_length_samples, hop_length=hop_length_samples)
    input_mag = np.abs(stft)
    input_phase = np.angle(stft)
    output_mag = []
    output_phase = [input_phase[:, 0]]
    i = 0
    while papping_function(i) < number_of_frames:
        input_frame = papping_function(i)
        int_input_frame = int(input_frame)
        ratio = input_frame - int(int_input_frame)
        output_mag.append(input_mag[:, int_input_frame] * (1 - ratio) + input_mag[:, (int_input_frame + 1)] * ratio)
        output_phase.append(output_phase[-1] + (input_phase[:, int_input_frame + 1] - input_phase[:, int_input_frame]))
        i += 1

    output_mag = np.array(output_mag).T
    output_phase = np.array(output_phase[:-1]).T
    new_stft = output_mag * np.exp(1j * output_phase)
    new_audio = librosa.istft(new_stft, n_fft=n_fft, win_length=window_length_samples, hop_length=hop_length_samples)
    return new_audio


if __name__ == '__main__':

    plot = False

    # ------------------------
    # question 1
    # ------------------------

    # a ----------------------
    # ii. Sampling rate is 48000 (I exported the wav file from Audacity)
    original_audio = load_audio_file('recording.wav')

    # b ----------------------
    resampled_audio_32 = resample_audio(48000, 32000, original_audio)

    # c ----------------------
    resampled_audio_16_1 = np.array([resampled_audio_32[i] for i in range(len(resampled_audio_32)) if i % 2 == 0],
                                    dtype=np.float32)
    resampled_audio_16_2 = resample_audio(32000, 16000, resampled_audio_32)

    # e ----------------------
    Plotter(resampled_audio_32, sr=32000).plot_all_plots() if plot else None
    Plotter(resampled_audio_16_1, sr=16000).plot_all_plots() if plot else None
    sf.write('resampled_audio_16_1.wav', resampled_audio_16_1, 16000)
    Plotter(resampled_audio_16_2, sr=16000).plot_all_plots() if plot else None
    sf.write('resampled_audio_16_2.wav', resampled_audio_16_2, 16000)

    # ------------------------
    # question 2
    # ------------------------

    # a ----------------------
    noise_original_audio = load_audio_file('stationary_noise.wav')
    noise_resampled_audio_16 = resample_audio(48000, 16000, noise_original_audio)

    # b ----------------------
    noise_and_signal_16_1 = resampled_audio_16_1 + noise_resampled_audio_16[:len(resampled_audio_16_1)]
    noise_and_signal_16_2 = resampled_audio_16_2 + noise_resampled_audio_16[:len(resampled_audio_16_2)]

    sf.write('noise_and_signal_16_2.wav', noise_and_signal_16_2, 16000)

    # c ----------------------
    Plotter(resampled_audio_16_2, sr=16000).plot_all_plots() if plot else None
    Plotter(noise_resampled_audio_16, sr=16000).plot_all_plots() if plot else None

    noise_and_signal_plotter = Plotter(noise_and_signal_16_2, sr=16000) if plot else None

    noise_and_signal_plotter.plot_all_plots() if plot else None

    # ------------------------
    # question 3
    # ------------------------

    # a ----------------------
    noise_threshold = -10
    noise_and_signal_plotter.plot_energy_and_rms(noise_threshold, plot_rms=False, title="Q3.a.i") if plot else None
    # vocal_samples = audio > noise_threshold
    # b ----------------------
    noise_reduction_signal, noise_audio = spectral_subtraction(noise_and_signal_16_2, noise_threshold)

    # c ----------------------
    Plotter(noise_reduction_signal, sr=16000).plot_all_plots() if plot else None
    Plotter(noise_audio, sr=16000).plot_all_plots() if plot else None
    sf.write('noise_reduction_signal.wav', noise_reduction_signal, 16000)

    # ------------------------
    # question 4
    # ------------------------

    # a ----------------------
    desired_rms = 0.1  # -20 DB
    noise_threshold = -10
    agc_signal, scaling_factors = auto_gain_control(resampled_audio_16_2, noise_threshold, desired_rms, hop_length_sec=0.01)
    Plotter(agc_signal, sr=16000).plot_all_plots() if plot else None
    sf.write('agc_signal.wav', noise_reduction_signal, 16000)

    seconds = np.array(range(len(resampled_audio_16_2))) / 16000
    hop_length_samples = int(0.01 * 16000)
    window_length_samples = int(0.02 * 16000)
    seconds = seconds[:-window_length_samples: hop_length_samples]
    if plot:
        plt.plot(seconds, scaling_factors)
        plt.xlabel('Time [secs]')
        plt.ylabel('Scaling Factors')
        plt.title('Scaling Factors VS Time')
        plt.show()

    # ------------------------
    # question 5
    # ------------------------

    # a ----------------------
    mapping_function = lambda x: x * 1.5
    speed_factor_audio = time_stretching_algorithm(resampled_audio_16_2, mapping_function)
    Plotter(speed_factor_audio, sr=16000).plot_all_plots() if plot else None
    sf.write('speed_factor_audio.wav', speed_factor_audio, 16000)

    print("Done")

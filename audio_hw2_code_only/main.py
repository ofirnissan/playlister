import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy
import os
from plotter import Plotter
from matplotlib import pyplot as plt
import matplotlib
from dtw import DTW
import copy


def load_audio_file(audio_file_path: str):
    audio, sr = librosa.load(audio_file_path, sr=48000)
    print(f"Sampling rate is: {sr}")
    return audio


def resample_audio(original_sr, output_sr, audio):
    secs = len(audio) / original_sr  # number of seconds in the audio
    samples = int(secs * output_sr)  # number of samples to resample
    return np.array(scipy.signal.resample(audio, samples), dtype=np.float32)


def resample_recording_files(recording_directory, out_path_directory, sample_frequency=16000):
    resampled_audio_signals_dict = {}
    for speaker_name in os.listdir(recording_directory):
        resampled_audio_signals_dict[speaker_name] = dict()
        sub_dir_path = os.path.join(recording_directory, speaker_name)
        sub_dir_out_path = os.path.join(out_path_directory, speaker_name)
        os.makedirs(sub_dir_out_path, exist_ok=True)
        for file_name in os.listdir(sub_dir_path):
            audio = load_audio_file(os.path.join(sub_dir_path, file_name))
            resampled_audio = resample_audio(48000, sample_frequency, audio)
            digit_name = file_name.split(".")[0]
            sf.write(os.path.join(sub_dir_out_path, f'{digit_name}.wav'), resampled_audio, sample_frequency)
            resampled_audio_signals_dict[speaker_name][int(digit_name)] = resampled_audio
    return resampled_audio_signals_dict


def plot_mel_spectogram(audio_signals_dict, sr=16000, win_length_sec=0.025, hop_length_sec=0.01, n_filters=80):
    matplotlib.use('Agg')

    os.makedirs('mel_spectogram_plots', exist_ok=True)
    os.makedirs('spectogram_plots', exist_ok=True)
    for speaker_name in audio_signals_dict:
        mel_spec_out_dir_path = os.path.join('mel_spectogram_plots', speaker_name)
        spec_out_dir_path = os.path.join('spectogram_plots', speaker_name)
        os.makedirs(mel_spec_out_dir_path, exist_ok=True)
        os.makedirs(spec_out_dir_path, exist_ok=True)

        for digit in audio_signals_dict[speaker_name]:
            plt.figure()
            plot_title = f'Mel Spectogram - {speaker_name}, digit - {digit}'
            plotter_obj = Plotter(audio_signals_dict[speaker_name][digit], sr=sr, win_length_sec=win_length_sec,
                                  hop_length_sec=hop_length_sec)
            plotter_obj.plot_mel_spectogram(title=plot_title, n_mels=n_filters)
            plt.savefig(os.path.join(mel_spec_out_dir_path, f'{digit}.png'))

            plt.figure()
            plot_title = f'Spectogram - {speaker_name}, digit - {digit}'
            plotter_obj.plot_spectogram(title=plot_title)
            plt.savefig(os.path.join(spec_out_dir_path, f'{digit}.png'))


def get_mel_spec_dict(audio_signals_dict, sr=16000, win_length_sec=0.025, hop_length_sec=0.01, n_filters=80):
    window_length_samples = int(win_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)
    n_fft = window_length_samples

    mel_spec_dict = dict()
    for speaker in resampled_audio_signals_dict:
        mel_spec_dict[speaker] = dict()
        for digit in resampled_audio_signals_dict[speaker]:
            mel_signal = librosa.feature.melspectrogram(y=audio_signals_dict[speaker][digit], sr=sr,
                                                        win_length=window_length_samples, hop_length=hop_length_samples,
                                                        n_fft=n_fft, n_mels=n_filters)
            mel_spec_dict[speaker][digit] = mel_signal
    return mel_spec_dict


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


if __name__ == '__main__':
    # ------------------------
    # question 1
    # ------------------------

    # b ----------------------
    resampled_audio_signals_dict = resample_recording_files('recordings', 'sampled_recordings', 16000)

    # c ----------------------
    class_representative = 'Yael_female'

    training_set = ['Einat_female', 'Gal_male', 'Orna_female', 'Itzik_male']
    evaluation_set = ['Ofir_male', 'Lesli_female', 'Iddo_male', 'Dafna_female']

    # ------------------------
    # question 2
    # ------------------------

    if not os.path.exists('mel_spectogram_plots'):
        plot_mel_spectogram(resampled_audio_signals_dict, sr=16000, win_length_sec=0.025, hop_length_sec=0.01,
                            n_filters=80)

    # ------------------------
    # question 3
    # ------------------------

    # c + d + e ------------------------
    mel_spec_dict = get_mel_spec_dict(resampled_audio_signals_dict, sr=16000, win_length_sec=0.025, hop_length_sec=0.01,
                                      n_filters=80)

    dtw_obj = DTW(class_representative, mel_spec_dict)

    distance_matrix = dtw_obj.construct_distance_matrix_multi_speaker(training_set, title='Training Set Distance Matrix')
    distance_matrix = dtw_obj.construct_distance_matrix_multi_speaker(evaluation_set, title='Validation Set Distance Matrix')

    # f ------------------------
    training_accuracy = dtw_obj.compute_confusion_matrix(training_set, title='Training Set Confusion Matrix')
    print(f"Training accuracy: {training_accuracy}")

    # g ------------------------
    validation_accuracy = dtw_obj.compute_confusion_matrix(evaluation_set, title='Validation Set Confusion Matrix')
    print(f"Validation accuracy: {validation_accuracy}")

    # plot distance matrix for each speaker

    # for speaker in training_set + evaluation_set:
    #     distance_matrix_one_speaker = dtw_obj.construct_distance_matrix_one_speaker(speaker)
    #     closest_matches = [np.argmin(distance_matrix_one_speaker[i]) for i in range(len(distance_matrix_one_speaker))]
    #     closest_matches_distance = [np.min(distance_matrix_one_speaker[i]) for i in range(len(distance_matrix_one_speaker))]

    # h -----------------------------

    for speaker in resampled_audio_signals_dict:
        for digit in resampled_audio_signals_dict[speaker]:
            new_audio, sf = auto_gain_control(resampled_audio_signals_dict[speaker][digit],
                                              -10, 0.01, win_length_sec=0.025, hop_length_sec=0.01)
            resampled_audio_signals_dict[speaker][digit] = new_audio
    mel_spec_dict = get_mel_spec_dict(resampled_audio_signals_dict, sr=16000, win_length_sec=0.025, hop_length_sec=0.01,
                                      n_filters=80)
    dtw_obj = DTW(class_representative, mel_spec_dict)

    training_accuracy = dtw_obj.compute_confusion_matrix(training_set, title='Training Set Confusion Matrix after AGC + Normalizing', normalize_length=True)
    print(f"Training accuracy after AGC + Normalizing: {training_accuracy}")
    validation_accuracy = dtw_obj.compute_confusion_matrix(evaluation_set, title='Validation Set Confusion Matrix after AGC + Normalizing', normalize_length=True)
    print(f"Validation accuracy after AGC + Normalizing: {validation_accuracy}")

    print("Done")

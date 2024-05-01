import numpy as np
from matplotlib import pyplot as plt


class Distance:
    def __init__(self, mel_spec_1, mel_spec_2):
        self.mel_spec_1 = np.log(np.abs(mel_spec_1))
        self.mel_spec_2 = np.log(np.abs(mel_spec_2))

    def calc(self, i, j):
        return np.sum((self.mel_spec_1[:, i] - self.mel_spec_2[:, j])**2)


class DTW:
    def calculate_dtw_distance(self, mel_spec_1, mel_spec_2, normalize_length=True):
        
        distance_obj = Distance(mel_spec_1, mel_spec_2)

        dtw_matrix = np.zeros((mel_spec_1.shape[1], mel_spec_2.shape[1]))
        dtw_matrix[0, 0] = distance_obj.calc(0, 0)
        for i in range(1, mel_spec_1.shape[1]):
            dtw_matrix[i, 0] = distance_obj.calc(i, 0) + dtw_matrix[i - 1, 0]
        for j in range(1,mel_spec_2.shape[1]):
            dtw_matrix[0, j] = distance_obj.calc(0, j) + dtw_matrix[0, j - 1]

        for i in range(1, mel_spec_1.shape[1]):
            for j in range(1, mel_spec_2.shape[1]):
                dtw_matrix[i, j] = distance_obj.calc(i, j) + np.min([dtw_matrix[i-1, j-1], dtw_matrix[i, j-1],
                                                                     dtw_matrix[i-1, j]])

        dtw_cost = dtw_matrix[-1, -1]
        if normalize_length:
            dtw_cost = dtw_cost / (mel_spec_1.shape[1] + mel_spec_2.shape[1])
        return dtw_cost

    def construct_distance_matrix_one_speaker(self, speaker, show=True, normalize_length=False):
        distance_matrix = np.zeros((len(self.mel_spec_dict[speaker]), len(self.mel_spec_dict[speaker])))
        for speaker_digit in self.mel_spec_dict[speaker]:
            for class_representative_digit in self.mel_spec_dict[self.class_representative_name]:
                distance_matrix[speaker_digit, class_representative_digit] = self.calculate_dtw_distance(
                    speaker, speaker_digit, class_representative_digit, normalize_length=normalize_length)
        if show:
            plt.matshow(distance_matrix, cmap='viridis_r')
            plt.title(speaker)
            plt.xlabel(f'DB Digit')
            plt.ylabel(f'Speaker Digit')
            plt.colorbar()
            plt.show()

        return distance_matrix

    def construct_distance_matrix_multi_speaker(self, speakers_list, show=True, title='distance_matrix',
                                                normalize_length=False):
        assert len(speakers_list) > 0
        distance_matrix = np.zeros((len(speakers_list)*10, len(self.mel_spec_dict[speakers_list[0]])))
        for i, speaker in enumerate(speakers_list):
            speaker_matrix = self.construct_distance_matrix_one_speaker(speaker, show=False, normalize_length=normalize_length)
            distance_matrix[10 * i: 10 * i + 10, :] = speaker_matrix
        if show:
            plt.matshow(distance_matrix, cmap='viridis_r')
            plt.xlabel('Digit')
            plt.ylabel('Speakers')
            plt.title(title)
            plt.colorbar()
            plt.show()

        return distance_matrix

    def compute_confusion_matrix(self, speakers_list, show=True, title='confusion matrix', normalize_length=False):
        confusion_matrix = np.zeros((10, 10))
        for speaker in speakers_list:
            distance_matrix = self.construct_distance_matrix_one_speaker(speaker, show=False,
                                                                         normalize_length=normalize_length)
            predicted_labels = np.argmin(distance_matrix, axis=1)
            for i in range(len(predicted_labels)):
                confusion_matrix[predicted_labels[i], i] += 1
        if show:
            plt.matshow(confusion_matrix)
            plt.xlabel('Expected')
            plt.ylabel('Predicted')
            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix[0])):
                    text = plt.text(j, i, f"{confusion_matrix[i, j]:10.2f}",
                                   ha="center", va="center", color="w")

            plt.title(title)
            plt.colorbar()
            plt.show()

        accuracy = np.sum([confusion_matrix[i, i] for i in range(len(confusion_matrix))]) / np.sum(confusion_matrix)
        return accuracy
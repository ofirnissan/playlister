import numpy as np
from matplotlib import pyplot as plt
import pickle


# collapse function
def B(input_string, blank_simbol="^"):
    output_string = ""
    i = 0
    while i < len(input_string):
        if input_string[i] != blank_simbol:
            output_string += input_string[i]
            while i < len(input_string) - 1 and input_string[i] == input_string[i + 1]:
                i += 1
        i += 1
    return output_string


def forward_pass(prediction_matrix, integer_sequence, blank_integer=2):
    y_matrix = np.zeros((len(integer_sequence), prediction_matrix.shape[0]), dtype=np.float32)
    y_matrix_force_alignment = np.zeros((len(integer_sequence), prediction_matrix.shape[0]), dtype=np.float32)
    y_matrix[0, 0] = prediction_matrix[0, integer_sequence[0]]
    y_matrix[1, 0] = prediction_matrix[0, integer_sequence[1]]
    y_matrix_force_alignment[0, 0] = prediction_matrix[0, integer_sequence[0]]
    y_matrix_force_alignment[1, 0] = prediction_matrix[0, integer_sequence[1]]
    for t in range(1, y_matrix.shape[1]):
        y_matrix[0, t] = prediction_matrix[t, integer_sequence[0]] * y_matrix[0, t - 1]
        y_matrix_force_alignment[0, t] = prediction_matrix[t, integer_sequence[0]] * y_matrix_force_alignment[0, t - 1]

    for t in range(1, y_matrix.shape[1]):
        for s in range(max(1, y_matrix.shape[0] - (y_matrix.shape[1] - t) * 2), y_matrix.shape[0]):
            if integer_sequence[s] == blank_integer or s < 2:
                y_matrix[s, t] = prediction_matrix[t, integer_sequence[s]] * (
                            y_matrix[s - 1, t - 1] + y_matrix[s, t - 1])
                y_matrix_force_alignment[s, t] = prediction_matrix[t, integer_sequence[s]] * max(
                    y_matrix_force_alignment[s - 1, t - 1], y_matrix_force_alignment[s, t - 1])
            else:
                y_matrix[s, t] = prediction_matrix[t, integer_sequence[s]] * (
                            y_matrix[s - 2, t - 1] + y_matrix[s - 1, t - 1] + y_matrix[s, t - 1])
                y_matrix_force_alignment[s, t] = prediction_matrix[t, integer_sequence[s]] * max(
                    y_matrix_force_alignment[s - 2, t - 1], y_matrix_force_alignment[s - 1, t - 1],
                    y_matrix_force_alignment[s, t - 1])

    return y_matrix, y_matrix_force_alignment


def backtrack_and_find_the_most_probable_path(y_matrix_force_alignment, integer_sequence, blank_integer=2):
    assert y_matrix_force_alignment.shape[0] > 1 and y_matrix_force_alignment.shape[1] > 1
    i = y_matrix_force_alignment.shape[0] - 1
    j = y_matrix_force_alignment.shape[1]
    most_probable_path = []
    while i > 0 and j > 0:

        if y_matrix_force_alignment[i, j-1] > y_matrix_force_alignment[i-1, j-1]:
            if integer_sequence[i] != blank_integer and i > 1 and y_matrix_force_alignment[i-2, j - 1] > y_matrix_force_alignment[i, j - 1]:
                most_probable_path.append(integer_sequence[i-2])
                i = i - 2
            else:
                most_probable_path.append(integer_sequence[i])
        else:

            if integer_sequence[i] != blank_integer and i > 1 and y_matrix_force_alignment[i-2, j - 1] > y_matrix_force_alignment[i - 1, j - 1]:
                most_probable_path.append(integer_sequence[i-2])
                i = i - 2
            else:
                most_probable_path.append(integer_sequence[i - 1])
                i = i - 1
        j = j-1

    while j > 0:
        most_probable_path.append(integer_sequence[i])  # i = 0
        j = j - 1

    most_probable_path.reverse()
    return most_probable_path


def display_matrix(sequence, matrix, title='', text=True, log_scale=False):
    display_labels = [s for s in sequence]
    fig, ax = plt.subplots()
    if log_scale:
        matrix = np.log(matrix)
    im = ax.imshow(matrix)
    ax.set_yticks(np.arange(len(matrix)), labels=display_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Label')
    if text:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                text = ax.text(j, i, f"{matrix[i, j]:10.2f}",
                               ha="center", va="center", color="w")
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    # ------------------------
    # question 5
    # ------------------------

    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00

    label_mapping = {0: 'a', 1: 'b', 2: '^'}
    reversed_label_mapping = {label_mapping[key]: key for key in label_mapping}

    display_matrix('ab^', pred.T, title='pred matrix')

    sequence = '^a^b^a^'
    integer_sequence = [reversed_label_mapping[token] for token in sequence]
    y_matrix, y_matrix_force_alignment = forward_pass(pred, integer_sequence, blank_integer=2)
    display_matrix(sequence, y_matrix, title="CTC's forward pass")

    # ------------------------
    # question 6
    # ------------------------
    print("--------------------------Q6--------------------------")

    display_matrix(sequence, y_matrix_force_alignment, title='force alignment matrix')
    most_probable_path_integer = backtrack_and_find_the_most_probable_path(
        y_matrix_force_alignment, integer_sequence, blank_integer=2)
    most_probable_path = "".join([label_mapping[i] for i in most_probable_path_integer])
    most_probable_path_probability = max(y_matrix_force_alignment[-1,-1], y_matrix_force_alignment[-2,-1])
    print(f"The most probable sequence (before collapse) is: {most_probable_path}; the probability for this path is: {most_probable_path_probability}")

    # ------------------------
    # question 7
    # ------------------------

    print("--------------------------Q7--------------------------")
    a = pickle.load(open('force_align.pkl', 'rb'))
    sequence = "^" + "^".join(a['text_to_align']) + "^"
    label_mapping = a['label_mapping']
    pred = a['acoustic_model_out_probs']
    reversed_label_mapping = {label_mapping[key]: key for key in label_mapping}
    integer_sequence = [reversed_label_mapping[token] for token in sequence]
    y_matrix, y_matrix_force_alignment = forward_pass(pred, integer_sequence, blank_integer=2)
    display_matrix(sequence, y_matrix, title="CTC's forward pass Q7", text=False, log_scale=True)
    display_matrix(sequence, y_matrix_force_alignment, title='force alignment matrix Q7', text=False, log_scale=True)
    most_probable_path_integer = backtrack_and_find_the_most_probable_path(
        y_matrix_force_alignment, integer_sequence, blank_integer=2)
    most_probable_path = "".join([label_mapping[i] for i in most_probable_path_integer])
    most_probable_path_probability = max(y_matrix_force_alignment[-1,-1], y_matrix_force_alignment[-2,-1])
    print(f"The most probable sequence (before collapse) is: {most_probable_path}; the probability for this path is: {most_probable_path_probability}")

    print("Done")

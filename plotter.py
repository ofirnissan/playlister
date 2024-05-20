import matplotlib.pyplot as plt
import numpy as np
import librosa


class Plotter:
    def __init__(self, audio, sr=16000, win_length_sec=0.02, hop_length_sec=0.01):
        self.audio = audio
        self.sr = sr
        self.window_length_samples = int(win_length_sec * sr)
        self.hop_length_samples = int(hop_length_sec * sr)

    def _add_audio_plot(self, ax):
        seconds = np.array(range(len(self.audio))) / self.sr
        ax.plot(seconds, self.audio)
        ax.set_title('Audio')
        ax.set_xlabel('Time [secs]')
        ax.set_ylabel('Amplitude')

    def _draw_pitch(self, ax):
        import parselmouth
        sound = parselmouth.Sound(self.audio, sampling_frequency=self.sr)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        ax.plot(pitch.xs(), pitch_values + 500, 'o', markersize=1.5, color='w')
        ax.plot(pitch.xs(), pitch_values + 500, 'o', markersize=0.75)

    def _add_spectogram_plot(self, fig, ax, draw_pitch=True):
        n_fft = self.window_length_samples
        stft = librosa.stft(self.audio, n_fft=n_fft, win_length=self.window_length_samples,
                            hop_length=self.hop_length_samples)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(stft_db, sr=self.sr, y_axis='linear', x_axis='time', ax=ax,
                                       hop_length=self.hop_length_samples)
        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        ax.set_xlabel('Time [secs]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Spectogram')
        if draw_pitch:
            self._draw_pitch(ax)

    def _add_mel_spectogram_plot(self, fig, ax, n_mels=80):
        n_fft = self.window_length_samples
        mel_signal = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, win_length=self.window_length_samples,
                                                    hop_length=self.hop_length_samples, n_fft=n_fft, n_mels=n_mels)

        power_to_db = librosa.power_to_db(mel_signal, ref=np.max)
        img = librosa.display.specshow(power_to_db, sr=self.sr, y_axis='mel', x_axis='time', ax=ax,
                                       hop_length=self.hop_length_samples)

        fig.colorbar(img, format='%+2.0f dB', ax=ax)
        ax.set_xlabel('Time [secs]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Mel-Spectogram')

    def _add_energy_and_rms_plot(self, fig, ax, threshold=None, plot_rms=True):
        if not fig:
            fig, ax = plt.subplots()

        energy = np.array([
            sum(abs(self.audio[i:i + self.window_length_samples] ** 2))
            for i in range(0, len(self.audio) - self.window_length_samples, self.hop_length_samples)
        ])
        energy = 10 * np.log10(energy)
        t = librosa.frames_to_time(range(len(energy)), sr=self.sr, hop_length=self.hop_length_samples)

        rms = (np.sum(self.audio ** 2) / len(self.audio)) ** 0.5
        rms = 20 * np.log10(rms)

        ax.plot(t, energy, label='energy')
        if plot_rms:
            ax.plot([0, t[-1]], [rms, rms], label='rms')

        if threshold is not None:
            ax.plot([0, t[-1]], [threshold, threshold], label='threshold')
        ax.set_title('Energy and RMS')
        ax.set_xlabel('Time [secs]')
        ax.set_ylabel('Energy [dB]')
        ax.legend()

    def plot_all_plots(self):
        fig, axs = plt.subplots(2, 2)
        self._add_audio_plot(axs[0, 0])
        self._add_spectogram_plot(fig,axs[0, 1])
        self._add_mel_spectogram_plot(fig, axs[1, 1])
        self._add_energy_and_rms_plot(fig, axs[1, 0])
        plt.show()

    def plot_energy_and_rms(self, threshold=None, plot_rms=True, title='Energy and RMS'):
        fig, ax = plt.subplots()
        self._add_energy_and_rms_plot(fig, ax, threshold=threshold, plot_rms=plot_rms)
        plt.title(title)
        plt.show()

    def plot_mel_spectogram(self, title='Mel Spectogram', n_mels=80):
        fig, ax = plt.subplots()
        self._add_mel_spectogram_plot(fig, ax, n_mels=n_mels)
        plt.title(title)
        plt.show()

    def plot_spectogram(self, title='Spectogram'):
        fig, ax = plt.subplots()
        self._add_spectogram_plot(fig, ax)
        plt.title(title)
        plt.show()

    def plot_audio(self, title='Audio'):
        fig, ax = plt.subplots()
        self._add_audio_plot(ax)
        plt.title(title)
        plt.show()

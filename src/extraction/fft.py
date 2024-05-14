import numpy as np
from scipy.signal import welch

from src.utils import get_env_variable


def dominant_frequency(signal, sampling_rate):
    N = len(signal)
    frequencies = np.fft.rfftfreq(N, 1 / sampling_rate)
    fft_values = np.abs(np.fft.rfft(signal))
    return frequencies[np.argmax(fft_values)]


def spectral_energy(signal, sampling_rate):
    fft_values = np.abs(np.fft.rfft(signal))
    return np.sum(fft_values ** 2)


def spectral_entropy(signal, sampling_rate):
    fft_values = np.abs(np.fft.rfft(signal))
    normalized_spectrum = fft_values / np.sum(fft_values)
    return -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))


def spectral_centroid(signal, sampling_rate):
    N = len(signal)
    frequencies = np.fft.rfftfreq(N, 1 / sampling_rate)
    fft_values = np.abs(np.fft.rfft(signal))
    return np.sum(frequencies * fft_values) / np.sum(fft_values)


def spectral_spread(signal, sampling_rate):
    N = len(signal)
    frequencies = np.fft.rfftfreq(N, 1 / sampling_rate)
    fft_values = np.abs(np.fft.rfft(signal))
    centroid = spectral_centroid(signal, sampling_rate)
    return np.sqrt(np.sum(((frequencies - centroid) ** 2) * fft_values) / np.sum(fft_values))


def band_power(signal, sampling_rate):
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=251)
    return np.sum(Pxx[(f >= 0.5) & (f <= 5)])


def peak_magnitudes_and_frequencies(signal, sampling_rate, top_n_peaks=3):
    N = len(signal)
    frequencies = np.fft.rfftfreq(N, 1 / sampling_rate)
    fft_values = np.abs(np.fft.rfft(signal))
    peak_indices = np.argsort(fft_values)[-top_n_peaks:]
    peak_magnitudes = fft_values[peak_indices].tolist()
    peak_frequencies = frequencies[peak_indices].tolist()

    peaks = {}
    peaks.update({f"peak_{i}_magnitude": peak_magnitudes[i] for i in range(top_n_peaks)})
    peaks.update({f"peak_{i}_frequency": peak_frequencies[i] for i in range(top_n_peaks)})
    return peaks


def spectral_flux(signal, sampling_rate):
    fft_values = np.abs(np.fft.rfft(signal))
    return np.sqrt(np.sum(np.diff(fft_values) ** 2))


def extract_fft_features(signal_column, sampling_rate=get_env_variable('RESAMPLE_RATE_HZ')):
    """Extract FFT related features from signal data."""
    assert sampling_rate, "Sampling rate not set in environment variables."

    features = {'dominant_frequency': dominant_frequency(signal_column, sampling_rate),
                'spectral_energy': spectral_energy(signal_column, sampling_rate),
                'spectral_entropy': spectral_entropy(signal_column, sampling_rate),
                'spectral_centroid': spectral_centroid(signal_column, sampling_rate),
                'spectral_spread': spectral_spread(signal_column, sampling_rate),
                'band_power': band_power(signal_column, sampling_rate),
                'spectral_flux': spectral_flux(signal_column, sampling_rate)}
    features.update(peak_magnitudes_and_frequencies(signal_column, sampling_rate))
    return features

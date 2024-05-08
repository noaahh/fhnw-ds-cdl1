import numpy as np
from scipy.signal import welch
from enum import Enum

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
    peak_magnitudes = fft_values[peak_indices]
    peak_frequencies = frequencies[peak_indices]
    return {
        "peak_magnitudes": peak_magnitudes.tolist(),
        "peak_frequencies": peak_frequencies.tolist()
    }

def spectral_flux(signal, sampling_rate):
    fft_values = np.abs(np.fft.rfft(signal))
    return np.sqrt(np.sum(np.diff(fft_values) ** 2))

class FFTMetrics(Enum):
    DOMINANT_FREQUENCY = dominant_frequency
    SPECTRAL_ENERGY = spectral_energy
    SPECTRAL_ENTROPY = spectral_entropy
    SPECTRAL_CENTROID = spectral_centroid
    SPECTRAL_SPREAD = spectral_spread
    BAND_POWER = band_power
    PEAK_MAGNITUDES_AND_FREQUENCIES = peak_magnitudes_and_frequencies
    SPECTRAL_FLUX = spectral_flux

def calculate_metrics(signal, sampling_rate, metrics_list):
    results = {}
    for metric in metrics_list:
        results[metric.__name__] = metric(signal, sampling_rate)
    return results

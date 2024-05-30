import numpy as np
import pywt
import scipy.signal as signal


def _calc_butterworth_filter(order, cutoff, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_butterworth_filter(data, order, cutoff, sampling_rate):
    b, a = _calc_butterworth_filter(order, cutoff, sampling_rate)
    return signal.filtfilt(b, a, data)


def apply_wavelet_denoising(data, wavelet='db4', level=1, mode='per'):
    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=level)
    threshold = np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet, mode=mode)
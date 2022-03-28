import numpy as np


class SpectralPerterbator:  # haha
    def __init__(self, spectra=None):
        self.target_spectrra = spectra

    def train(self, signals, skip=None, N=100):
        """learns the N average highst FFT coefficients, from list in positional argument 1, skipping signals[j] where skip[j]=True"""
        if skip is None:
            skip = ["False"] * len(signals)
        elif len(skip) != len(signals):
            raise ValueError("dimension mismathc")
        H = np.zeros((len(signals), int(len(signals[0]) / 2 + 1)))
        for j, s in enumerate(signals):
            if not skip[j]:  # wana look at the spectra of the noise
                H[j, 0::] = np.fft.rfft(s)
        self.target_spectra_means = np.mean(H, 1)
        self.target_spectra_varience = np.var(H, 1)
        ranks = np.argsort(np.mean(H, 1))
        self.target_spectra_varience[
            ranks[0:-N]
        ] = 0  # makes the perterbation 0 for everything but ther top N means

    def proc(self, signal, c=0.05):
        """perterbs the fourier coefficients of list of signals by variance 1 mean 0 normal random proportionally by c default value 0.05"""
        out = []
        for s in signal:
            S = np.array(np.ravel(s))
            n = len(S)
            out.append(
                S
                + np.fft.irfft(
                    np.random.normal(self.target_spectra_means, target_spectra_varience)
                )
            )

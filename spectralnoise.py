import numpy as np


class SpectralPerterb:  # haha
    def __init__(self, means=None, variace=None, N=None):
        if means is not None and variance is not None:
            self.target_spectra_variance
            self.target_spectra_mean = means

    def fit(self, signals, skip=None):
        """calculates and saves the means and variances of the discrete cosine transform of the iterable object of input signals, skipping the elements where skip==True"""
        N = len(signals[0])
        Nsig = len(signals)
        if skip is None:
            skip = [False] * Nsig
        elif len(skip) != Nsig:
            raise ValueError("dimension mismathc")
        H = np.zeros((Nsig, int(N / 2 + 1)))
        for j, s in enumerate(signals):
            if not skip[j]:
                H[j, :] = (2 / np.sqrt(N)) * np.fft.rfft(s).real
        self.target_spectra_means = np.mean(H, 1)
        self.target_spectra_variance = np.sqrt(np.var(H, 1)) * np.sqrt(Nsig)
        self.N = N
        self.H = H
        self.Nsig = Nsig

    def perterb(self, signal, c=0.05):
        """perterbs the fourier coefficients of an iterable list/array/tupple of signals(as numpy arrays)
        proportional to c with random samples from a normal distribution for each fourier coefficient
        with the mean and variance in self.target_spectra_variance and self.target_spectra_mean. It is a statistically-speaking, a unitary operation"""
        out = signal.copy() * (1 - c)
        for j, o in enumerate(out):
            out[j] += self.synth(o.size) * c
        return out

    def synth(self, n=None):
        """outputs a inverse-fft reconstructed signal randomly from the known means and variances, specified length defeaulting to training length"""
        if n is None:
            n = self.N
        return (np.sqrt(n) / 2) * np.fft.irfft(
            np.random.normal(
                self.target_spectra_means.real,
                self.target_spectra_variance.real,
            ),
            n=n,
        )

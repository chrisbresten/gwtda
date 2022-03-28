import numpy as np


class SpectralPerterbator:  # haha
    def __init__(self, spectra_means=None, specta_variance=None, N=None, mean=None):
        """doesnt need any initial input, but the state(means and variances as an array_like) can be set so that a saved or externally calculated or synthetic distribution can be used"""
        self.target_spectra_means = spectra_means
        self.target_spectra_variance = specta_variance
        self.N = N
        if mean is None:
            self.u = 1
        else:
            self.u = mean

    def fit(self, signals, skip=None, M=None):
        """learns the M average highst FFT coefficients, from list in positional argument 1, skipping signals[j] where skip[j]=True"""
        N = len(signals[0])
        Nsig = len(signals)
        u = signals.mean()
        if skip is None:
            skip = [False] * Nsig
        elif len(skip) != Nsig:
            raise ValueError("dimension mismathc")
        if M is None:
            M = N
        H = np.zeros((Nsig, int(N/2)+1), dtype=np.complex128)
        for j, s in enumerate(signals):
            if not skip[j]:  # wana look at the spectra of the noise
                H[j, 0::] = np.fft.rfft(s, norm="ortho")
        self.target_spectra_means = np.mean(H, 1)
        self.target_spectra_variance = np.var(H, 1)
        self.N = N
        self.u = u
        ranks = np.argsort(np.mean(H.abs, 1))
        self.target_spectra_variance[
            ranks[0:-M]
        ] = 0  # makes the perterbation 0 for everything but ther top N means

    def perterb(self, signal, c=0.05):
        """perterbs the fourier coefficients of an iterable list/array/tupple of signals(as numpy arrays)
        proportional to c=0.05(default) with random samples from a normal distribution for each fourier coefficient
        with the mean and variance in self.target_spectra_variance and self.target_spectra_mean. It is a statistically-speaking, a unitary operation"""
        out = []
        for s in signal:
            S = np.array(np.ravel(s))
            n = len(S)
            out.append(
                S * (1 - c)
                + c
                * np.fft.irfft(
                    np.random.normal(
                        self.target_spectra_means.real,
                        np.sqrt(self.target_spectra_variance.real),
                    )
                    + np.random.normal(
                        self.target_spectra_means.imag,
                        np.sqrt(self.target_spectra_variance).imag,
                    )
                    * np.complex128(complex(0, 1)),
                    n=n,
                    norm="ortho",
                )
            )
        return np.array(out)

    def predict(self, n=None):
        """outputs a inverse-fft reconstructed signal randomly from the known means and variances"""
        if n is None:
            n = self.N
        return self.u * np.fft.irfft(
            np.random.normal(
                self.target_spectra_means, np.sqrt(self.target_spectra_variance)
            ),
            n=n,
        )

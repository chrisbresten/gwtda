import numpy as np


class SpectralPerterb:  # haha
    def __init__(self):
        pass

    def fit(self, signals, skip=None):
        """learns the M average highst FFT coefficients, from list in positional argument 1, skipping signals[j] where skip[j]=True"""
        N = len(signals[0])
        Nsig = len(signals)
        if skip is None:
            skip = [False] * Nsig
        elif len(skip) != Nsig:
            raise ValueError("dimension mismathc")
        H = np.zeros((Nsig, int(N / 2 + 1)))
        for j, s in enumerate(signals):
            if not skip[
                j
            ]:  # wana look at the spectra of the noise, not the positive detected signals
                H[j, :] = (2 / np.sqrt(N)) * np.fft.rfft(s).real
        self.target_spectra_means = np.mean(H, 1)
        self.target_spectra_variance = np.sqrt(np.var(H, 1))*np.sqrt(Nsig)
        self.N = N
        self.H = H
        self.Nsig = Nsig

    def _perterb(self, signal, c=0.05):
        """perterbs the fourier coefficients of an iterable list/array/tupple of signals(as numpy arrays)
        proportional to c=0.05(default) with random samples from a normal distribution for each fourier coefficient
        with the mean and variance in self.target_spectra_variance and self.target_spectra_mean. It is a statistically-speaking, a unitary operation"""
        out = []
        n = self.N
        for j in range(self.Nsig):
            out.append(
                H[j, :]
                + np.fft.irfft(
                    np.random.normal(
                        self.target_spectra_means.real,
                        self.target_spectra_variance.real,
                    ),
                    n=n,
                )
            )
        return np.array(out)

    def predict(self, n=None):
        """outputs a inverse-fft reconstructed signal randomly from the known means and variances"""
        if n is None:
            n = self.N
        return (np.sqrt(n) / 2) * np.fft.irfft(
            np.random.normal(
                self.target_spectra_means.real,
                self.target_spectra_variance.real,
            ),
            n=n,
        )

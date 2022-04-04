import numpy as np

try:
    from gwtools import mismatch
    from gwtools import gwutils
except:
    print("gwtools not installed, cant calculate SNR")


class SpectralPerterb:  # haha
    def __init__(self, means=None, std=None, N=None):
        """can initialize with synthetic or saved state"""
        if means is not None and std is not None:
            self.stds = std
            self.means = means

    def fit(self, signals, skip=None):
        """calculates and saves the means and stds of the discrete cosine
        transform coefficients of the iterable object of input signals,
        skipping the elements where skip==True"""
        N = len(signals[0])
        Nsig = len(signals)
        if skip is None:
            skip = [False] * Nsig
        elif len(skip) != Nsig:
            raise ValueError("dimension mismathc")
        H = np.zeros((Nsig, int(N / 2 + 1)))
        for j, s in enumerate(signals):
            if not skip[j]:
                H[j, :] = (1 / np.sqrt(N)) * np.fft.rfft(s)
        self.means = np.mean(H, 0)
        self.stds = np.std(H, 0)
        self.N = N
        self.H = H
        self.Nsig = Nsig

    def perterb(self, signals, c=0.05):
        """perterbs the fourier coefficients of an iterable list/array/tupple
        of signals(as numpy arrays) proportional to c with random samples from
        a normal distribution for each fourier coefficient with the mean and
        std in self.target_spectra_std and self.target_spectra_mean.
        It is a statistically-speaking, a unitary operation"""
        out = []
        for s in signals:
            out.append(
                np.sqrt(self.N)
                * np.fft.irfft(
                    (1 / np.sqrt(self.N))
                    * np.fft.rfft(s)
                    * np.random.normal(
                        np.ones(np.shape(self.means)),
                        (self.stds * (c / 2)),
                    )
                )
            )
        return out

    def perterb_white(self, signals, c=0.05):
        """perterbs the fourier coefficients of an iterable list/array/tupple
        of signals(as numpy arrays) proportional to c with random samples from
        a normal distribution for each fourier coefficient with the mean and
        std in self.target_spectra_std and self.target_spectra_mean.
        It is a statistically-speaking, a unitary operation"""
        out = []
        for s in signals:
            out.append(
                np.sqrt(self.N)
                * np.fft.irfft(
                    (1 / np.sqrt(self.N))
                    * np.fft.rfft(s)
                    * np.random.normal(1, c / 2, int(self.N / 2 + 1))
                )
            )
        return out

    def synth(self, n=None, means=None, stds=None):
        """outputs a inverse-fft reconstructed signal randomly from the known
        means and stds, specified length defeaulting to training length"""
        if means is None:
            means = self.means
        if stds is None:
            stds = self.stds

        if n is None:
            n = self.N
        return (np.sqrt(n)) * np.fft.irfft(
            np.random.normal(
                means,
                stds,
            )
        )

    def synth_white(self, n=None):
        """synths white noise scaled according to whats been fit"""
        return self.synth(
            n=n,
            means=np.mean(self.means) * np.ones(np.shape(self.means)),
            stds=np.mean(self.stds),
        )

    def snr(self, signal, signal_with_noise, freq=4096):
        """calculate snr of resulting signal based off of noise statistics used to generate it"""
        N = len(signal)
        dt = 1 / freq
        WaveformLength = dt * N
        freqs, hFreq = gwutils.freqDomainWaveform(signal, dt)
        freqs_noise, nFreq = gwutils.freqDomainWaveform(signal_with_noise, dt)
        df = 1 / WaveformLength
        Sn = np.pad(
            (np.sqrt(N)) * self.stds ** 2.0,
            (0, N - len(self.stds)),
            mode="symmetric",
        )
        numerator = np.real(
            mismatch.inner_complex(
                hFreq, nFreq, psd=Sn, df=df, ligo=True, pos_freq=False
            )
        )
        denominator = np.sqrt(
            mismatch.inner(hFreq, hFreq, psd=Sn, df=df, ligo=True, pos_freq=False)
        )
        match_filter_snr = numerator / denominator
        return match_filter_snr

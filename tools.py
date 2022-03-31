import __main__
import numpy as np
from matplotlib import pyplot as plt

try:
    from gwtools import mismatch
    from gwtools import gwutils
except:
    print("gwtools not installed, cant calculate SNR")


def serialize(x):
    """recursive function that converts nested arrays to lists and the numpy
    numeric data types to native python floats to make the structure json
    serializable, so that it can be dumped to json. input is iterable output is
    python list"""
    out = []
    for k in x:
        try:
            if len(k) > 0:
                out.append(serialize(list(k)))
        except TypeError:
            out.append(float(k))
    return out


def snr_whitenoise(signal, signal_with_noise, R, length=1):
    WaveformLength = length
    dt = WaveformLength / len(signal)
    freqs, hFreq = gwutils.freqDomainWaveform(signal, dt)
    freqs_noise, nFreq = gwutils.freqDomainWaveform(signal_with_noise, dt)
    df = 1 / WaveformLength
    noise_std = R
    Sn = 2.0 * dt * (noise_std ** 2.0) / WaveformLength
    numerator = np.real(
        mismatch.inner_complex(hFreq, nFreq, psd=Sn, df=df, ligo=True, pos_freq=False)
    )
    denominator = np.sqrt(
        mismatch.inner(hFreq, hFreq, psd=Sn, df=df, ligo=True, pos_freq=False)
    )
    match_filter_snr = numerator / denominator
    return match_filter_snr


def savestuff(end=False):
    if end:
        part = ""
    else:
        part = "_part"
    np.save(
        __main__.outfile + part,
        (
            ___main__.loadfile,
            ___main__.Npad,
            ___main__.Ndattotal,
            ___main__.ncoeff,
            ___main__.xsig,
            ___main__.bettiout,
            ___main__.pdout,
            ___main__.swout,
            ___main__.yout,
        ),
    )


def plot_signals(filen, show=True):
    g = np.load(filen, allow_pickle=True)
    for j, k in enumerate(g[0][1:30]):
        if g[1][j][0] == 1:
            plt.plot(k, "g")
        else:
            plt.plot(k, "r")
    if show:
        plt.show()

import numpy as np
from spectralnoise import SpectralPerterb

sp = SpectralPerterb()


def ligo_noise():
    """noise from interferometer observtions"""
    gwpre = np.load("datasample/segmentedLIGO2sec.npy", allow_pickle=True)
    y_init = np.concatenate(tuple(gwpre[1]), 0)
    signals = np.concatenate(tuple(gwpre[0]), 0)
    out = []
    for j, y in enumerate(y_init):
        if not y:  # if not signal-bearing window
            out.append(signals[j])
    return out


def chop_ligo_noise(V, n, ncoef):
    """adds ncoef*ligonoise to  V  and chops by random insertion into n points
    of noise, partially synthetic by random perterbation selective spetrally to
    the data as observed"""
    out = []
    for s in V:
        out.append(chop(s, n))
    return sp.perterb(out, c=ncoef)


def chop_white_noise(V, n, ncoef):
    """adds kr*whitenoise to  V  and chops by random insertion into n points of white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    out = []
    for s in V:
        out.append(chop(s, n))
    return sp.perterb_white(out, c=ncoef)


def chop(signal, nchop):
    """chops out a fixed size random position portion to use, providing
    variable position without artifacts and difficulties  and discontinuities
    of padding"""
    N = len(signal)
    cut = np.random.randint(nchop)
    return signal[cut : (N - nchop + cut)]


def fit(Nchop):
    lnoise = ligo_noise()
    clnoise = []
    for l in lnoise:
        clnoise.append(chop(l, Nchop))
    sp.fit(clnoise)


legend = {"ligo": chop_ligo_noise, "white": chop_white_noise}

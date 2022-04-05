import numpy as np
from spectralnoise import SpectralPerterb


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
    p = chop(V, n)
    return sp.perterb(p, c=ncoef)


def chop_white_noise(V, n, ncoef):
    """adds kr*whitenoise to  V  and chops by random insertion into n points of white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    p = chop(V, n)
    return sp.perterb_white(p, c=ncoef)


def chop(signal, nchop):
    """chops out a fixed size random position portion to use, providing
    variable position without artifacts and difficulties  and discontinuities
    of padding"""
    N = signal.size
    cut = np.random.randint(nchop)
    return signal[cut : (nchop - cut)]


sp = SpectralPerterb()
lnoise = ligo_noise()
sp.fit(lnoise)

legend = {"ligo": sp.perterb, "white": sp.perterb_white}

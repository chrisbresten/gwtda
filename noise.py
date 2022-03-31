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


def pad_ligo_noise(V, n, ncoef):
    """adds ncoef*ligonoise to  V  and pads by random insertion into n points of ligo noise
    which is scaled by factor ncoef from mean 0 var 1 nominal distribution"""
    N = V.size
    p = pad(V, n)
    return sp.perterb(p, c=ncoef)


def pad_white_noise(V, n, ncoef):
    """adds kr*whitenoise to  V  and pads by random insertion into n points of white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    N = V.size
    p = pad(V, n)
    return sp.perterb_white(p, c=ncoef)


def pad(signal, npad):

    N = signal.size
    cut = np.random.randint(npad)
    out = np.concatenate((np.zeros((cut,)), signal, np.zeros((npad - cut,))))
    return out


sp = SpectralPerterb()
lnoise = ligo_noise()
sp.fit(lnoise)

legend = {"ligo": pad_ligo_noise, "white": pad_white_noise}

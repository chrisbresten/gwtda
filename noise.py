import numpy as np
from spectralnoise import SpectralPerterb


def ligo_noise():
    """noise from interferometer observtions"""
    name = "ligo"
    gwpre = np.load("datasample/segmentedLIGO2sec.npy", allow_pickle=True)
    y_init = np.concatenate(tuple(gwpre[1]), 0)
    signals = np.concatenate(tuple(gwpre[0]), 0)
    out = []
    for j, y in enumerate(y_init):
        if not y:
            out.append(signals[j])
    return out


def pad_ligo_noise(V, n, kr):
    """adds kr*ligonoise to  V  and pads by random insertion into n points of ligo noise white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    N = V.size
    cut = np.random.randint(n)
    randw = sp.synth(N + n)
    out = np.concatenate((np.zeros(( cut,)), V, np.zeros(( n - cut,))))
    return out + randw * kr


def pad_white_noise(V, n, kr):
    """adds kr*whitenoise to  V  and pads by random insertion into n points of white noise white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    cut = np.random.randint(n)
    rand1 = np.random.randn(cut)
    rand2 = np.random.randn(n - cut)
    randm = np.random.randn(V.size)
    out = np.concatenate((rand1 * kr, V + kr * randm, rand2 * kr))
    return out


sp = SpectralPerterb()
lnoise = ligo_noise()
sp.fit(lnoise)

legend = {"ligo": pad_ligo_noise, "white": pad_white_noise}

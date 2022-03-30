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


def pad_ligo_noise(V, n, kr):
    """adds kr*ligonoise to  V  and pads by random insertion into n points of ligo noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    N = V.size
    cut = np.random.randint(n)
    randw = sp.synth(N + n)
    out = np.concatenate((np.zeros((cut,)), V, np.zeros((n - cut,))))
    scale = np.ones(np.shape(out))
    scale[0:cut] = 1 / kr
    scale[(N + cut) : :] = 1 / kr
    return out + randw * scale * kr


def pad_white_noise(V, n, kr):
    """adds kr*whitenoise to  V  and pads by random insertion into n points of white noise
    which is scaled by factor kr from mean 0 var 1 nominal distribution"""
    N = V.size
    cut = np.random.randint(n)
    randw = sp.synth_white(N + n)
    out = np.concatenate((np.zeros((cut,)), V, np.zeros((n - cut,))))
    scale = np.ones(np.shape(out))
    scale[0:cut] = 1 / kr
    scale[(N + cut) : :] = 1 / kr
    return out + randw * scale * kr


sp = SpectralPerterb()
lnoise = ligo_noise()
sp.fit(lnoise)

legend = {"ligo": pad_ligo_noise, "white": pad_white_noise}

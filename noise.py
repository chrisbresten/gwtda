
import numpy as np



def gw_surroate(Nsamp=Nsamp):
    """loads a saved sample of surrogate model(truncated fourier series) gravitational wave signatures from binary collisions. Generated with GWtools"""
    synthetic_sigs = "datasample/data1samp.npy"
    gw = np.load(synthetic_sigs)
    Nsig = len(gw["signal_present"])
    signals = gw["data"]
    return signals

def ligo():
    """noise from interferometer observtions"""
    name = "ligo"
    gwpre = np.load("datasample/segmentedLIGO2sec.npy", allow_pickle=True)
    y_init = np.concatenate(tuple(gwpre[1]), 0)
    signals = np.concatenate(tuple(gwpre[0]), 0)
    out = []
    for j,y in enumerate(y_init):
        if not y:
            out.append(signals[j])
    return out 

def whitenoise(N=100)
    out = []
    for j in range(n):
        

def padrand(V, n, kr):
    """pads a signal white noise scaled by factor kr"""
    cut = np.random.randint(n)
    rand1 = np.random.randn(cut)
    rand2 = np.random.randn(n - cut)
    out = np.concatenate((rand1 * kr, V, rand2 * kr))
    return out



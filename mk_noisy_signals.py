import signal_sources
import noise

import sys
from embeddings import slidend, dimred3
import numpy as np

Ndattotal = 1000
Npad = 1000
Nfactor = 1
ncoeff = 0
try:
    signal_type = sys.argv[1]
    noise_type = sys.argv[2]
    if len(sys.argv) > 3:
        Ndattotal = int(sys.argv[3])
except (IndexError, ValueError):
    SystemExit(
        f"Usage {sys.argv[0]} <signal type gw | chirps | ligo > <noise ligo | whitenoise> <Ndata> \n whre (optional) Ndata is total elements to  create"
    )

try:
    signal_source_fun = signal_sources.legend[signal_type]
except:
    sigt = str(list(signal_sources.legend.keys()))
    SystemExit(f"unknown signal type {signal_type} not in {sigt}")
try:
    nfun = noise.legend[noise_type]
except:
    noiset = str(list(noise.legend.keys()))
    SystemExit(f"unknown noise type {noise_type} not in {noiset}")

signals = signal_source_fun()
Nsig = len(signals)
N = len(signals[0])
ncoeff = 0.4  # coefficient that noise is scaled by
p = 0.5  # proportion of elements with a signal present
Nwindow = 200  # sliding window size


y = np.zeros((Ndattotal, Nsig))
psignals = []
x = []
xsig = []
ybin = []
pcloud = []

for j in range(Ndattotal):
    yn = np.random.randint(Nsig)
    if np.random.rand() < p:
        signal = signals[yn]
        ybin.append([1, 0])
    else:
        signal = np.zeros(signals[yn].shape)
        ybin.append([0, 1])
    for kk in range(Nsig):
        y[j, kk] = int(yn == kk)  # turn to binary vector len n for n sig
    psignals.append(nfun(signal, Npad, ncoeff))

for j, rawsig in enumerate(psignals):
    slidez = slidend(rawsig / np.abs(rawsig).max(), Nwindow)
    threed = dimred3(slidez)
    xsig.append(rawsig.copy())
    pcloud.append(threed.copy())
    if j % (int(Ndattotal / 100)) == 0:
        jj = int((100 * (j / Ndattotal)))
        print("  %s   %s    " % (jj, "%"), end="\r")


outfile = "%s_signal_sliding_windowN%d_%d.npy" % (
    signal_type + "_" + noise_type + "_" + str(ncoeff),
    Ndattotal,
    np.random.randint(5, 5000),
)
np.save(
    outfile,
    (xsig, ybin, pcloud, Npad, ncoeff, Nwindow, Ndattotal, N, signals, Nsig, y),
)

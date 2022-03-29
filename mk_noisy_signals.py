import signals
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
    SystemExit(        f"Usage {sys.argv[0]} <signal type gw | chirps | ligo > <noise ligo | whitenoise> <Ndata> \n whre (optional) Ndata is total elements to  create"    )

try:
    fun = signals.legend[signal_type]
    nfun = noise.legend[noise_type]
except:
    sigt = str(list(signals.legend.keys()))
    SystemExit(f"unknown signal type {signal_type} not in {sigt}")

signals = fun()
Nsig = len(signals)
N = len(signals[0])
ncoeff = 0.4
x = []
xsig = []
ybin = []
pcloud = []
maxs = 0.0
k = 0
p = 0.5
Nwindow = 200  # sliding window size
y = np.zeros((Ndattotal, Nsig))
psignals = []

for j in range(Ndattotal):
    yn = np.random.randint(Nsig)
    if np.random.rand() < p:
        signal = signals[yn]
        ybin.append([1, 0])
    else:
        signal = signals[yn] * 0
        ybin.append([0, 1])
    for kk in range(Nsig):
        y[j, kk] = int(yn == kk)  # turn to binary vector len n for n sig
    psignals.append(nfun(signal, Npad, ncoeff))

for j,rawsig in enumerate(psignals):
    slidez = slidend(rawsig / np.abs(rawsig).max(), Nwindow)
    threed = dimred3(slidez)
    xsig.append(rawsig.copy())
    pcloud.append(threed.copy())
    if j % (int(Ndattotal / 100)) == 1:
        jj = int((100 * j) / Ndattotal)
        print("%s            " % (jj,))


outfile = "%s_signal_sliding_windowN%d_%d.npy" % (
    name,
    Ndattotal,
    np.random.randint(5, 5000),
)
np.save(
    outfile,
    (xsig, ybin, pcloud, Npad, ncoeff, Nwindow, Ndattotal, N, signals, Nsig, Nsamp, y),
)

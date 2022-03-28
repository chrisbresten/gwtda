from spectralnoise import SpectralPerterbator
import sys
from embeddings import slidend, dimred3, padrand
import numpy as np

# make synthetic soscillatory signals with additive whitenoise and padded with white noise
# save the raw signals and the sliding window embedding dimensionally reduced to 3D
Ndattotal = 1000
Nsamp = 1  # downsample by this factor unless overwritten
Npad = 1000
Nfactor = 1
ncoeff = 0
try:
    signal_type = sys.argv[1]
    if len(sys.argv) > 2:
        Ndattotal = int(sys.argv[2])
except (IndexError, ValueError):
    SystemExit(
        f"Usage {sys.argv[0]} <signal type gw | chirps | ligo | whitenoise> <Ndata> \n whre (optional) Ndata is total elements to create"
    )


def syntheticchirps():
    """set of 4 locally periodic genetic wave packets for testng and visualization"""
    N = 2000
    dx = np.linspace(-1, 1, N)
    signals = [
        0 * dx,
        (np.sin(15 * np.pi * dx))
        * (np.sin(32 * np.pi * dx) + np.cos(12 * np.pi * dx) + np.sin(79 * np.pi * dx)),
        (np.exp(-3 * dx))
        * (np.sin(9 * np.pi * dx) + np.cos(15 * np.pi * dx) + np.sin(63 * np.pi * dx)),
        (-(dx ** 2) + 3) * np.sin(30 * np.pi * dx),
    ]
    Nsig = len(signals)
    for j in np.arange(Nsig - 1) + 1:
        signals[j] = signals[j] / (signals[j].max())
    return signals


def gwwhitenoise(Nsamp=Nsamp):
    """loads up synthetic GW signals made with surrogate model, noiseless"""
    synthetic_sigs = "datasample/data1samp.npy"
    global Nfactor
    Nfactor = 10 ** (-19)
    gw = np.load(synthetic_sigs)
    Nsig = len(gw["signal_present"])
    signals = gw["data"]
    return signals

if signal_type == "chirps":
    name = "chirps"
    signals = syntheticchirps()
    ncoeff = 0.5
elif signal_type == "w":
    name = "whitenoise"
    gw = np.load(synthetic_sigs)
    signals = np.zeros(np.shape(gw["data"]))
    ncoeff = 1
elif signal_type == "ligo":
    name = "ligo"
    gwpre = np.load("datasample/segmentedLIGO2sec.npy", allow_pickle=True)
    ncoeff = 0.05
    y_init = np.concatenate(tuple(gwpre[1]), 0)
    signals = np.concatenate(tuple(gwpre[0]), 0)
    spe = SpectralPerterbator()
    spe.train(signals,skip=y_init)
else:
    Nsamp = 4
    ncoeff = 0.5
    name = "gw"
    signals = gwwhitenoise()
print(name)
Nsig = len(signals)
N = len(signals[0][0:-1:Nsamp])

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
if signal_type == "ligo":
    _psignals = []
    for j in range(Ndattotal):
        yn = np.random.randint(Nsig)
        _psignals.append(signals[yn])
        if y_init[yn]:
            ybin.append([1, 0])
        else:
            ybin.append([0, 1])
    psignals = spe.proc(_psignals,ncoeff)

else:
    for j in range(Ndattotal):
        yn = np.random.randint(Nsig)
        if np.random.rand() < p:
            signal = signals[yn][0:-1:Nsamp]
            ybin.append([1, 0])
        else                     :
            signal = signals[yn][0:-1:Nsamp] * 0
            ybin.append([0, 1])
        noise = ncoeff * np.random.randn(N)
        for kk in range(Nsig):
            y[j, kk] = int(yn == kk)  # turn to binary vector len n for n sig
        psignals.append(padrand(signal + noise, Npad, ncoeff))

for j,rawsig in enumerate(psignals):
    slidez = slidend(rawsig / np.abs(rawsig).max(), Nwindow)
    threed = dimred3(slidez)
    xsig.append(rawsig.copy())
    pcloud.append(threed.copy())
    if j % (int(Ndattotal / 100)) == 1:
        print("%3d %" % (int((100 * j) / Ndattotal),),end='')


outfile = "%s_signal_sliding_windowN%d_%d.npy" % (
    name,
    Ndattotal,
    np.random.randint(5, 5000),
)
np.save(
    outfile,
    (xsig, ybin, pcloud, Npad, ncoeff, Nwindow, Ndattotal, N, signals, Nsig, Nsamp, y),
)

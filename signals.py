import numpy as np

# make synthetic soscillatory signals with additive whitenoise and padded with white noise
# save the raw signals and the sliding window embedding dimensionally reduced to 3D


def chirps(Npts=2000):
    """set of 4 locally periodic genetic wave packets for testng and
    visualization. Nothing special, just generic items for example
    classification of wave packets."""
    dx = np.linspace(-1, 1, Npts)
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


def gw_surroate():
    """loads a saved sample of surrogate model(truncated fourier series)
    gravitational wave signatures from binary collisions. Generated with
    GWtools"""
    synthetic_sigs = "datasample/data1samp.npy"
    gw = np.load(synthetic_sigs)
    Nsig = len(gw["signal_present"])

    _signals = gw["data"]
    out = []
    Nsamp = 4;
    for s in _signals:
        out.append(s[0:-1:Nsamp])
    return out


def ligo_events():
    """several identified ligo observations"""
    name = "ligo"
    gwpre = np.load("datasample/segmentedLIGO2sec.npy", allow_pickle=True)
    y_init = np.concatenate(tuple(gwpre[1]), 0)

    signals = np.concatenate(tuple(gwpre[0]), 0)
    out = []
    for j, y in enumerate(y_init):
        if y:
            out.append(signals[j])
    return out

legend = {"chirps": chirps, "gw": gw_surroate, "ligo": ligo_events}


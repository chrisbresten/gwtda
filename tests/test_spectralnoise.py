import spectralnoise
import numpy as np

g = spectralnoise.SpectralPerterb()
h = np.load("ligo_signal_sliding_windowN4000_4863.npy", allow_pickle=True)
g.fit(h[0][1:100])
for k in h[0][1:100]:
    print(np.linalg.norm(k))
    print(np.linalg.norm(g.synth(n=8096)))
    print(np.linalg.norm(g.perterb(k)))
    print(np.linalg.norm(g.perterb(k, c=0.5)))

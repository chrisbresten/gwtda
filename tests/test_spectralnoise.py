import spectralnoise
import numpy as np

norm = np.linalg.norm
rands = [np.random.randn(4096) for r in range(20)]
g = spectralnoise.SpectralPerterb()
# h = np.load("ligo_signal_sliding_windowN4000_4863.npy", allow_pickle=True)
g.fit(rands)
print("synth_white", norm(g.synth_white()))
print("synth", norm(g.synth()))
# randp = g.perterb(rands)
# print("perp", norm(g.perterb(randp, c=0.5)))
# print("perp white", norm(g.perterb_white(randp, c=0.5)))
# for k in range(5):
#    #print(norm(randp[k]))
print(g.means)

import noise
import signal_sources

g = spectralnoise.SpectralPerterb()
l = noise.ligo_noise()
cl = signal_sources.ligo_events()
g.fit(l)
print("synth", norm(g.synth()))
print("synth_white", norm(g.synth_white()))
ln = g.perterb(cl, c=0.5)
lnl = g.perterb(cl, c=0.05)
lg = g.perterb_white(l, c=0.5)
c = g.perterb(cl, c=0.5)

for li in range(10):
    print("noise", norm(l[li]))
    print("signal",norm(cl[li]) )
    print("sig+nopise 0.5", norm(ln[li]))
    print("sig+0.05 noise", norm(lnl[li]))
    print("whitenoise + sig 0.5", norm(lg[li]))
    print("snr", g.snr(cl[li], c[li]))


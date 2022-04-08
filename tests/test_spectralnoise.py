from tools import tp
from termcolor import colored
import spectralnoise
import numpy as np

norm = np.linalg.norm


@tp
def test_mag():
    g = spectralnoise.SpectralPerterb()
    rands = [np.random.randn(4096) for r in range(20)]
    g.fit(rands)
    rm =    norm(g.synth_white())/norm(g.synth()) 
    return((rm < 1.05) and (rm > 0.95))

import noise
import signal_sources

g = spectralnoise.SpectralPerterb()
l = np.array(noise.ligo_noise())
cl = signal_sources.ligo_events().copy()
g.fit(l)

@tp
def test_mag__perterb(): 
    ln = g.perterb(cl, c=0.5)
    lnl = g.perterb(cl, c=0.05)
    lg = g.perterb_white(l, c=0.5)
    clp = g.perterb(cl, c=0.5)
    print(norm(lg))
    clp = g.perterb(cl, c=0.5)
    print(norm(clp))
    clp = g.perterb(cl, c=0.5)
    print(norm(clp))
    clp = g.perterb(cl, c=0.5)
    print(norm(clp))
    clp = g.perterb(cl, c=0.5)
    print(norm(clp))

    print(norm(ln))
    rm =  norm(ln)/norm(lnl) 
    return rm < 1.4 and rm > 0.9
@tp
def testsnr_gauss():
    n1 = np.random.randn(l[0].size)*np.mean(l)
    n2 = np.random.randn(l[0].size)*np.mean(l)
    return  (np.abs(g.snr(n1,n2))<0.01) and (np.abs(g.snr(n2,g.synth_white())<0.01))



testsnr_gauss()

test_mag__perterb()
test_mag()

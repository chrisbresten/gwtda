from matplotlib import pyplot as plt
import spectralnoise
import numpy as np
import noise 
import signal_sources
norm = np.linalg.norm
sigs = signal_sources.ligo_events()
noise.fit(0)
sig = sigs[0]
sign = []

for k in range(10):
    print(norm(sigs[k]))
    for j in range(9):
        sign = noise.sp.perterb(sigs,c=0.1*j)
        print(noise.sp.snr(sigs[k],sign[k]))
        print(norm(sign[k]))
        plt.plot(sign[k])
plt.show()

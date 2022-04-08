from matplotlib import pyplot as plt
import spectralnoise
import numpy as np
import noise
import signal_sources
import embeddings

norm = np.linalg.norm
sigs = signal_sources.ligo_events()
noise.fit(0)
sig = sigs[0]
sign = []

for k in range(1):
    print(norm(sigs[k]))
    plt.figure()
    for j in range(4):
        sign = noise.sp.perterb(sigs, c=0.2 * j)
        ax0 = plt.subplot(212)
        ax0.scatter(
            np.linspace(0, 256, sign[k].size),
            sign[k],
            s=1,
            c=np.linspace(0, 256, sign[k].size),
            cmap="viridis",
        )
        slidez = embeddings.slidend(sign[k] / np.abs(sign[k]).max(), 200)
        threed = embeddings.dimred3(slidez)
        snr = noise.sp.snr(sigs[k], sign[k])
        print(snr)
        plt.title("snr " + str(snr))

        ax1 = plt.subplot(222, projection="3d")
        ax1.scatter(
            threed[:, 0],
            threed[:, 1],
            threed[:, 2],
            c=np.linspace(0, 256, len(threed)),
            cmap="viridis",
        )

        ax21 = plt.subplot(221, projection="3d")
        ax21.view_init(elev=80,azim=90)
        ax21.scatter(
            threed[:, 0],
            threed[:, 1],
            threed[:, 2],
            c=np.linspace(0, 256, len(threed)),
            cmap="viridis",
        )


        plt.savefig("knot" + str(j) + str(k) + ".eps")
        plt.close()

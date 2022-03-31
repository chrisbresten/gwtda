from tools import savestuff
import sys
import numpy as np
import embeddings

try:
    loadfile = sys.argv[1]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} signals_swindow_embedded.npy ")


Ksw = embeddings.sw_rep_embedding(Nrefs=50)

(
    xsig,
    y,
    pcloud,
    Npad,
    ncoeff,
    Nwindow,
    Ndattotal,
    N,
    signals,
    Nsig,
    sigind,
) = np.load(loadfile, allow_pickle=True)
outfile = "tda_" + loadfile[0:10] + str(np.random.randint(10000))
bettiout = []
pdout = []
swout = []
yout = []
for j, pp in enumerate(pcloud):
    p = np.array(pp)
    print(" %s    %s" % (int(100 * j / len(pcloud)), "%"), end="\r")
    b0, b1, pd0, pd1 = embeddings.mkbothvec_alpha(np.array(p), 25, np.sqrt(2), 50)
    kSWlap = Ksw.embed(p / np.linalg.norm(p, np.inf))
    bettiout.append((b0, b1))
    pdout.append((pd0, pd1))
    swout.append(kSWlap)
    yout.append(y[j])
    if (j > 1) and (j % int(Ndattotal / 10) == 0):
        savestuff()
        print("\n save %s percent" % (int(100 * j / N),))
savestuff(True)


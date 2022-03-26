import sys
import numpy as np
import embeddings


try:
    loadfile = sys.argv[1]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} signals_swindow_embedded.npy ")


# load data and do the tda components then save the outcome
Ksw = embeddings.kernel_ref_transform(Nrefs=50)

# load tjhe 3d poiont cloudsa
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
    Nsamp,
    sigind,
) = np.load(loadfile, allow_pickle=True)
outfile = "tda_" + loadfile[0:10] + str(np.random.randint(10000))
bettiout = []
pdout = []
swout = []
yout = []
for j, pp in enumerate(pcloud):
    p = np.array(pp)
    print(j / len(pcloud))
    b0, b1, pd0, pd1 = embeddings.mkbothvec_alpha(np.array(p), 25, np.sqrt(2), 50)
    kSWlap = Ksw.project(p / np.linalg.norm(p, np.inf))
    bettiout.append((b0, b1))
    pdout.append((pd0, pd1))
    swout.append(kSWlap)
    yout.append(y[j])
    if j % int(Ndattotal / 10 + 1) == 0:
        np.save(
            outfile + "_" + str(j),
            (
                loadfile,
                Npad,
                Ndattotal,
                ncoeff,
                Nsamp,
                xsig,
                bettiout,
                pdout,
                swout,
                yout,
            ),
        )
        print("save")


np.save(
    outfile + "_" + Ndattotal,
    (
        loadfile,
        Npad,
        Ndattotal,
        ncoeff,
        Nsamp,
        xsig,
        bettiout,
        pdout,
        swout,
        yout,
    ),
)

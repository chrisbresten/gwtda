import tensorflow as tf
import __main__
import numpy as np
from matplotlib import pyplot as plt

try:
    from gwtools import mismatch
    from gwtools import gwutils
except:
    print("gwtools not installed, cant calculate SNR")


def serialize(x):
    """recursive function that converts nested arrays to lists and the numpy
    numeric data types to native python floats to make the structure json
    serializable, so that it can be dumped to json. input is iterable output is
    python list"""
    out = []
    for k in x:
        try:
            if len(k) > 0:
                out.append(serialize(list(k)))
        except TypeError:
            out.append(float(k))
    return out



def savetda(end=False):
    if end:
        part = ""
    else:
        part = "_part"
    np.save(
        __main__.outfile + part,
        (
            __main__.loadfile,
            __main__.Npad,
            __main__.Ndattotal,
            __main__.ncoeff,
            __main__.xsig,
            __main__.bettiout,
            __main__.pdout,
            __main__.swout,
            __main__.yout,
        ),
    )


def plot_signals(filen, show=True):
    g = np.load(filen, allow_pickle=True)
    for j, k in enumerate(g[0][1:30]):
        if g[1][j][0] == 1:
            plt.plot(k, "g")
        else:
            plt.plot(k, "r")
    if show:
        plt.show()


def prep4Classifier(loadfile, embedi):
    modes = ["pd", "sw", "bv", "all"]
    """prepares the data for classification, input is file and embedding type, output is embedded signals and raw signals"""
    (
        filename_original,
        Npad,
        Ndattotal,
        ncoeff,
        xsig,
        bettiout,
        pdout,
        swout,
        yout,
    ) = np.load(loadfile, allow_pickle=True)
    y = []
    signals = []  # time domain raw signal
    xembed = []  # embeddings
    xconcat = []  # combined
    if embedi == "pd":
        daN = 100
    else:
        daN = 50
    Ndat = len(yout)
    maxsig = 0.0
    for xn in xsig:
        maxsig = max(maxsig, np.abs(xn).max())
    for j in range(Ndat):
        if embedi == modes[0]:  # pd
            b = pdout[j]
            xx = np.reshape(np.array(list(b[0]) + list(b[1])), (daN, 1))
        elif embedi == modes[1]:  # sw
            b = np.array(swout[j])
            daN = b.size
            xx = np.reshape(b, (daN, 1))
        elif embedi == modes[2]:  # bv
            b = bettiout[j]
            xx = np.reshape(np.array(list(b[0]) + list(b[1])), (daN, 1))
        elif embedi == modes[3]:  # all
            b = np.array(
                list(pdout[j][0])
                + list(pdout[j][1])
                + list(swout[j])
                + list(bettiout[j][0])
                + list(bettiout[j][1])
            )
            daN = b.size
            xx = np.reshape(b, (daN, 1))
        signals.append(np.reshape(xsig[j] / maxsig, (len(xsig[j]), 1)))
        # raw normalized signal
        xembed.append(xx / xx.max())
        y.append(yout[j])
        xconcat.append(
            np.concatenate(
                ((xx / xx.max()), np.reshape(xsig[j] / maxsig, (len(xsig[j]), 1)))
            )
        )
        # embedding
    return (np.array(xconcat), np.array(y), np.array(signals), np.array(xembed))


def loadfromjson(CONFIGJSON, serialweights):
    """loads a sequential model from json as string or dict, with accomidation
    for various user ineptitudes with regards to the need for consistent input
    types and data structure.  returns tensorflow model"""
    if type(CONFIGJSON) == str:
        _json_cfg = json.loads(CONFIGJSON)
    else:  # if type(CONFIGJSON)==dict:
        _json_cfg = CONFIGJSON
    try:
        json_cfg = _json_cfg["config"]
    except KeyError:
        json_cfg = _json_cfg
    try:
        model = tf.keras.models.Sequential.from_config(json_cfg)
    except:
        print(json_cfg)
        raise
    theshapes = []
    for l in model.layers:
        for w in l.weights:
            theshapes.append(tuple(w.shape.as_list()))
    theweights = []
    for j, s in enumerate(serialweights):
        theweights.append(np.reshape(s, theshapes[j]))
    model.set_weights(theweights)
    return model

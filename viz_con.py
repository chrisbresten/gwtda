from tools import serialize, loadfromjson, prep4Classifier, getmodel
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
import numpy as np

try:
    model, loadfile, cmdline_args = getmodel(sys.argv[1])
except:
    model, loadfile, cmdline_args = getmodel()
weights = model.get_weights()
embedtype = cmdline_args.split(" ")[-1]
(x, y, xsig, xema) = prep4Classifier(loadfile, embedtype)
if "run_CNN" in cmdline_args:
    xem = x
    #print(x.shape)
    raw = 1
else:
    raw = 0
    xem = xema
(a, c, b) = np.array(weights[0]).shape
convweights = np.array(weights[0])
convbias = np.array(weights[1])


def loadactiv(model, layer):
    model.compile()
    return model.layers[layer].call


if embedtype == "all":
    vlines = [50, 100, 150, 200, 175]

comment = [
    "p vector H_0",
    "p vector H_1",
    "sw embed",
    "betti-vector h0",
    "betti-vector h1",
]


elif embedtype == "pd":
    vlines = [50]
    comment=[ "p vector H_0",    "p vector H_1"]
elif embedtype == "bv":
    vlines = [25]
    comment =["betti-vector h0",    "betti-vector h1"]

    
if raw == 1:
    #    vlines = list(vlines)  + [np.sum(vlines)]
    comment = comment + ["raw signal"]


def mkplot(JK):
    fig = plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ze = np.reshape(xem[JK], (xem[JK].size,))
    re = ze.copy()
    re[(ze == 0)] = 1
    j = 1
    print(xema)
    img = []
    imgno = []
    # act0 = loadactiv(model,0)
    for j in range(b):
        if j < 5:
            print(convweights[:, 0, j])
        convN = convweights[:, 0, j].size
        c = np.convolve(ze, convweights[:, 0, j], mode="valid") * convbias[j]
        img.append(c)
        imgno.append(
            c / np.convolve(ze, np.ones(convweights[:, 0, j].shape), mode="valid")
        )
        # imgact.append(act0(img[-1]))
    aimg = np.array(img).reshape(len(img), len(img[0]))
    ax1.imshow(np.log10(np.abs(imgno) + 0.000000001))
    if y[JK][0] == 1:
        plt.title("signal present")
    else:
        plt.title("no signal")
    ax1.annotate(
        "normalized to input max, log",
        xy=(0, 0),
        xytext=(12, -12),
        rotation=0,
        va="center",
        annotation_clip=False,
    )

    ax2.imshow(img)

    ax2.annotate(
        "raw output",
        xy=(0, 0),
        xytext=(12, -12),
        rotation=0,
        va="center",
        annotation_clip=False,
    )

    ax2.set(xlim=[0, ze.size])
    ax1.set(xlim=[0, ze.size])
    # ax2.plot(act0(ze))
    ax3.set(xlim=[0, ze.size])
    ax3.plot(ze)
    for v in range(len(vlines)):
        ax1.axvline(vlines[v] - 2, color="red")
        ax2.axvline(vlines[v] - 2, color="red")
        ax3.axvline(vlines[v], color="red")
        ax3.annotate(
            comment[v],
            xy=(vlines[v] + 2, 0),
            xytext=(-10, 5),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="center",
            annotation_clip=False,
            # arrowprops=arrowprops,
        )
        plt.tight_layout()
    if raw == 1 or embedtype !='all':
        ax3.annotate(
            comment[-1],
            xy=(vlines[-1] + 2, 0),
            xytext=(20, 5),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="center",
            annotation_clip=False,
            # arrowprops=arrowprops,
        )


ze = []
ysig = 0
nosig = 0
K = 0
while ysig < 2 or nosig < 2:
    K = K + 1
    if y[K][1] == 1 and ysig < 2:
        ysig = ysig + 1
        mkplot(K)
        print(K)
    if y[K][0] == 0 and nosig < 2:
        nosig = nosig + 1
        mkplot(K)


plt.show()

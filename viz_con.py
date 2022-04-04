from tools import serialize, loadfromjson, prep4Classifier
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
import sys
import spsql

s = spsql.spsql()

SCHEMA = os.getenv("SCHEMA", "gw")

# s.curs.execute("select modelhash, savefile,loadfile, weights,cmdline_args from " + SCHEMA + ".runs order by test_accuracy desc limit 100")
s.curs.execute(
    "select modelhash, savefile,loadfile, weights,cmdline_args from "
    + SCHEMA
    + ".runs where weightshash=%s",
    (sys.argv[1],),
)

_weights = s.curs.fetchall()
(modelhash, savefile, loadfile, weights, cmdline_args) = _weights[0]
s.curs.execute(
    "select model_json from " + SCHEMA + ".models where modelhash=%s", (modelhash,)
)
_model_json = s.curs.fetchall()[0][0]

print(savefile, "  -  ", loadfile, "  -  ", cmdline_args)
print(np.array(weights["weights"][0]).shape)
print(np.ravel(weights["weights"][0]).shape)
(x, y, xsig, xema) = prep4Classifier(loadfile, cmdline_args.split(" ")[-1])
if "run_CNN" in cmdline_args:
    xem = x
    print(x.shape)
else:
    xem = xema
(a, c, b) = np.array(weights["weights"][0]).shape
convweights = np.array(weights["weights"][0])


def loadactiv(model_json, weights):
    if type(weights) is dict:
        weightsa = np.array(weights["weights"])
    model = loadfromjson(model_json, weightsa)
    model.compile()
    # returns array of functions
    return [model.layers[j].activation for j in range(len(model.layers))]


def mkplot(JK):
    ze = np.reshape(xem[JK], (xem[JK].size,))
    re = ze.copy()
    re[(ze == 0)] = 1
    csum = np.convolve(ze, convweights[:, 0, 0], mode="valid")
    j = 1
    print(xema)
    img = []
    for j in range(1, b):
        if j < 5:
            print(convweights[:, 0, j])
        c = np.convolve(ze, convweights[:, 0, j], mode="valid")
        csum = csum + c
        img.append(c / (re[2::]))
    plt.figure()
    aimg = np.array(img).reshape(len(img), len(img[0]))
    plt.imshow(np.log(np.abs(img) + 0.000001))
    return ze


ze = []
for K in range(5):
    ze.append(mkplot(K))
plt.figure()
for z in ze:
    plt.plot(z)


plt.show()

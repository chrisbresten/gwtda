from __future__ import division
import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
import sys


(
    filename_original,
    Npad,
    Ndattotal,
    ncoeff,
    Rcoeflist,
    xsig,
    bettiout,
    pdout,
    swout,
    swoutOrth,
    yout,
) = np.load(sys.argv[1], allow_pickle=True)
y = []
x = []
if sys.argv[2] == "pd":
    daN = 100
else:
    daN = 50
Ndat = len(yout)
for j in range(Ndat):
    if sys.argv[2] == "pd":
        b = pdout[j]
        xx = np.reshape(np.array(list(b[0]) + list(b[1])), (daN, 1))
    elif sys.argv[2] == "sw":
        b = np.array(swout[j])
        daN = b.size
        xx = np.reshape(b, (daN, 1))
    elif sys.argv[2] == "swo":
        b = np.array(swoutOrth[j])
        daN = b.size
        xx = np.reshape(b, (daN, 1))
    else:
        b = bettiout[j]
        xx = np.reshape(np.array(list(b[0]) + list(b[1])), (daN, 1))

    x.append((xx / xx.max()))
    y.append([yout[j][0], yout[j][1]])


Ntest = int(Ndat / 10)
y = np.array(y)
x = np.array(x)
x_train = x[0 : (Ndat - Ntest)]
x_test = x[(Ndat - Ntest + 1) : :]
y_train = y[0 : Ndat - Ntest]
y_test = y[Ndat - Ntest + 1 : :]

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(32, kernel_size=32, strides=1, input_shape=(daN, 1)),
        tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=4, padding="valid"
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Conv1D(64, kernel_size=3),
        tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=4, padding="valid"
        ),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Conv1D(64, kernel_size=8),
        tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=4, padding="valid"
            ),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),  
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32),  ,
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(2),  # ,activation=tf.nn.softmax)
    ]
)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=7)
k = model.evaluate(x_test, y_test)
print(k)
yp = model.predict(x_test)

with open(sys.argv[1][0:20] + option + ".json", "w") as json_file:
    json_file.write(model_json)

mkroc(y_test, yp)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def mkroc(ytest, ypred):
    # y_pred_keras = keras_model.predict(X_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(ytest.ravel(), ypred.ravel())
    auc_keras = auc(fpr_keras, tpr_keras)
    np.save(
        sys.argv[1].splot(".")[0] + "__" + +sys.argv[2] + ".npy",
        (
            fpr_keras,
            tpr_keras,
            auc_keras,
            thresholds_keras,
            Ndat,
            Ntest,
            ytest,
            ypred,
            option,
            x_test,
        ),
    )

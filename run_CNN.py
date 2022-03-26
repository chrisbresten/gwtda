import logrun
import tensorflow as tf
import numpy as np
import sys

modes = ["pd", "sw", "bv", "all"]
try:
    loadfile = sys.argv[1]
    embedi = sys.argv[2]
except IndexError:
    raise SystemExit(
        f"Usage: {sys.argv[0]} datafile_tda_features.npy <embedding> \n where <embedding> is one of{modes}"
    )

(
    filename_original,
    Npad,
    Ndattotal,
    ncoeff,
    Nsamp,
    xsig,
    bettiout,
    pdout,
    swout,
    yout,
) = np.load(loadfile, allow_pickle=True)
y = []
x = []
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
        daN = len(b)
        xx = np.reshape(b, (daN, 1))
    x.append(
        np.concatenate(
            ((xx / xx.max()), np.reshape(xsig[j] / maxsig, (len(xsig[j]), 1)))
        )
    )
    y.append([yout[j][0], yout[j][1]])


daN = len(x[0])

Ntest = int(Ndat / 10)
y = np.array(y)
x = np.array(x)
x_train = x[0 : (Ndat - Ntest)]
x_test = x[(Ndat - Ntest + 1) : :]
y_train = y[0 : Ndat - Ntest]
y_test = y[Ndat - Ntest + 1 : :]

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(daN / 8, kernel_size=8, strides=1, input_shape=(daN, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=16, strides=4, padding="valid"),
        tf.keras.layers.Conv1D(64, kernel_size=8, strides=1, input_shape=(daN, 1)),
        tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=4, padding="valid"
        ),  # data_format='channels_last'),
        tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, input_shape=(daN, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=4, strides=4, padding="valid"),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Conv1D(64, kernel_size=3),
        tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=4, padding="valid"
        ),  # data_format='channels_last'),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),  # activation=tf.nn.linear),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32),  # activation=tf.nn.linear),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(2),  # ,activation=tf.nn.softmax)
    ]
)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)
k = model.evaluate(x_test, y_test)
print(k)


yp = model.predict(x_test)
model_json = model.to_json()


logr = logrun.ModelLog(model_json)


# def mkroc(ytest, ypred):
# y_pred_keras = keras_model.predict(X_test).ravel()
ytest = y_test
ypred = yp
fpr_keras, tpr_keras, thresholds_keras = roc_curve(ytest.ravel(), ypred.ravel())
auc_keras = auc(fpr_keras, tpr_keras)
nameout = (
    "saveweights_"
    + sys.argv[0].split(".")[0].strip(" ")
    + "_WEIGHTS_"
    + str(Ndat)
    + embedi
    + "_"
    + str(np.random.randint(99999999))
    + ".npy"
)

np.save(
    nameout,
    (
        fpr_keras,
        tpr_keras,
        auc_keras,
        thresholds_keras,
        Ndat,
        Ntest,
        ytest,
        ypred,
        x_test,
        filename_original,
        sys.argv[0],
        ncoeff,
        Rcoeflist,
        model.get_config(),
        model.get_weights(),
    ),
)


def serialize(x):
    out = []
    for k in x:
        try:
            if len(k) > 0:
                out.append(serialize(list(k)))
        except TypeError:
            out.append(float(k))
    return out


weights = serialize(model.get_weights())
logr.logrun(
    k[1],
    Ntest,
    nameout,
    sys.argv[1],
    output={"auc_keras": auc_keras},
    weights={"weights": weights},
    notes="%s auc_keras: %f" % (str(k), auc_keras),
)

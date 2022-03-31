from tools import prep4Classifier
from sklearn.metrics import roc_curve, auc
from tools import serialize

try:
    import logrun
except ModuleNotFoundError:
    Warning("no database no logging/saving of results")
import tensorflow as tf
import numpy as np
import sys


try:
    loadfile = sys.argv[1]
    embedi = sys.argv[2]
except IndexError:
    raise SystemExit(
        f"Usage: {sys.argv[0]} datafile_tda_features.npy <embedding> \n where <embedding> is one of{modes}"
    )


(x, y, xsig, xem) = prep4Classifier(loadfile, embedi)
Ndat = len(y)
Ntest = int(Ndat / 10)
x_train = x[0 : (Ndat - Ntest)]
x_test = x[(Ndat - Ntest + 1) : :]
y_train = y[0 : Ndat - Ntest]
y_test = y[Ndat - Ntest + 1 : :]
daN = len(x[0])

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(32, kernel_size=15, strides=1, input_shape=(daN, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=7, strides=4, padding="valid"),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Conv1D(64, kernel_size=13),
        tf.keras.layers.MaxPooling1D(pool_size=3, strides=4, padding="valid"),
        tf.keras.layers.Dense(32, activation=tf.nn.softmax),
        tf.keras.layers.Conv1D(64, kernel_size=3),
        tf.keras.layers.MaxPooling1D(pool_size=3, strides=4, padding="valid"),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(2,activation=tf.nn.softmax),
    ]
)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)
k = model.evaluate(x_test, y_test)
print(k)


ypred = model.predict(x_test)
model_json = model.to_json()

try:
    logr = logrun.ModelLog(model_json)
except:
    Warning("Can't log model details")


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.ravel(), ypred.ravel())
auc_keras = auc(fpr_keras, tpr_keras)
nameout = (
    "run_CNN_"
    + sys.argv[0].split(".")[0].strip(" ")
    + str(Ndat)
    + embedi
    + "_"
    + str(np.random.randint(99999999))
    + ".npy"
)


weights = serialize(model.get_weights())
try:
    logr.logrun(
        k[1],
        Ntest,
        nameout,
        sys.argv[1],
        output={"auc_keras": auc_keras},
        weights={"weights": weights},
        notes="%s auc_keras: %f" % (str(k), auc_keras),
    )
except Exception as e:
    Warning(f"Can't log results: {e} \n saving to file... {nameout}")
    np.save(
        nameout,
        (
            fpr_keras,
            tpr_keras,
            auc_keras,
            thresholds_keras,
            Ndat,
            Ntest,
            y_test,
            ypred,
            x_test,
            sys.argv[0],
            model.get_config(),
            model.get_weights(),
        ),
    )

import sys

sys.path.insert('../')
from tools import serialize,loadfromjson,prep4Classifier
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
import spsql;
s = spsql.spsql()

SCHEMA =  os.getenv("SCHEMA","gw")
print(SCHEMA)
#s.curs.execute("select modelhash, savefile,loadfile, weights,cmdline_args from " + SCHEMA + ".runs order by test_accuracy desc limit 100")
s.curs.execute("select modelhash, savefile,loadfile, weights,cmdline_args from " + SCHEMA + ".runs where weightshash=%s",(sys.argv[1],))

_weights=s.curs.fetchall()
(modelhash, savefile, loadfile, weights,cmdline_args) = _weights[0]
s.curs.execute("select model_json from " + SCHEMA + ".models where modelhash=%s",(modelhash,))
#s.curs.execute("select model_json from " + SCHEMA + ".models where filehash=%s",('3f7b68a18069982d53b0b29d62e6d5da',))
try:
    _model_json = s.curs.fetchall()[0][0]
    model_json = json.loads(_model_json)
except TypeError as e:
    print("Json loaded by psycopg")
    model_json = _model_json
except:
    print("cant find the json for config: ",getattr("model_json",''))

typev = cmdline_args.split(' ')[-1]

(x, y, xsig, xem) = prep4Classifier(loadfile, typev)


model = loadfromjson(model_json,np.array(weights['weights']))
(modelhash, savefile, loadfile, weights,cmdline_args) = _weights[0]
#np.reshape(np.array(weights['weights'][0]),(3,32))

model.compile()
activations = model.layers[0].activation(x[0:3])
#convolutions are to let us look at a moving averafge for our eyes, pourely visual as reality is very choppy here and local average is what we want to see
for j in range(len(activations)): 
    if y[j][0]==1 :
        plt.plot(np.convolve(np.ravel(np.abs(activations[j][:])),np.ones((30,)),'valid'),'r',alpha=(j%10)/15+0.2) 
#    else :
#        plt.plot(np.convolve(np.ravel(activations[j][:]),np.ones((j,)),'valid'),'k',alpha=(j%10)/15+0.2) 
plt.show()

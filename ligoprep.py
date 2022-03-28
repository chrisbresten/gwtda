from __future__ import division
import numpy as np
from os import listdir
from os import sys
import re
from itertools import islice

hz = 4096  # samp rate
wsec = 2


def chopsec(dat):
    """chops up time series into overlapping segments"""
    win = hz * wsec
    hw = int(win / 2)
    N = int(np.floor(len(dat)/win)*win)
    start1 = np.arange(0, N , win)
    start2 = np.arange(hw, N-hw, win)
    start = np.sort(list(start1) + list(start2))
    dataout = []
    for s in start:
        dataout.append(dat[s : (s + win)])
    return dataout, start


def gettime(startindex, startwalltime, sigtime):
    """get the segments where the event is present, using the time of event as seen by data source."""
    timez = startindex / hz + startwalltime
    difft = timez - sigtime
    difftp = difft * (difft > 0).astype(np.int)
    out = (difftp < wsec) & (difftp > 0)
    print(difftp)
    print(out)
    return out


datapath = sys.argv[1]
rfiles = listdir(datapath)
# tids = []
names = []
datas = {}
# event times
sigtimes = {
    "GW170817": 1187008882.4,
    "GW170608": 1180922494.5,
    "GW170104": 1167559936.6,
    "GW151226": 1135136350.6,
    "GW150914": 1126259462.4,
    "GW170814": 1186741861.5,
}
timeinit = {}
times = []
ctimes = []
for r in rfiles:
    tid = re.findall("-(\d{10})-", r)
    with open(datapath + r) as myfile:
        head = list(islice(myfile, 3))
    for h in head:
        name = re.findall(" (GW.*?)_", h)
        if len(name) > 0:
            names.append(name[0])
            timeinit[name[0]] = int(tid[0])
            datas[name[0]] = np.loadtxt(datapath + r, comments="#")
x = []
y = []
timestarts = {}
for n in names:
    xsig, starts = chopsec(datas[n])
    ysig = gettime(starts, timeinit[n], sigtimes[n])
    x.append(xsig)
    y.append(ysig)
    timestarts[n] = timeinit[n] + starts
outfile = "segmented" + "LIGO" + str(wsec) + "sec"
print("saving to %s" % (outfile,))
np.save(outfile, (x, y, sigtimes, timeinit, datas, timestarts))

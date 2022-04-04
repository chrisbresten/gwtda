from signal_sources import chirps,ligo_events, gw_surroate
import numpy as np


print(chirps(Npts=4096))
print(gw_surroate())
print('ligo', ligo_events()[0][0:10],' ...')







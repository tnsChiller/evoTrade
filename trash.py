import numpy as np
import Utilities as util
import IndicatorsVectorized as ind
import time

hist = util.GetHist()
metrics = util.GetMetrics(hist)
popSize = 2
pop = util.StartMetricPopulation(metrics, popSize)

w, thr = pop[:, :metrics.shape[0]], pop[:, metrics.shape[0]:]
w = w.reshape((popSize, metrics.shape[0],1 ,1))
scr = w * metrics


c0, c1 = inds < pop[:, 0], pop[:, 1]
cLong = np.logical_and(np.all(c0, axis = 1), np.all(c1, axis = 1))
cPrev, cNow = cLong[:, :, :-1], cLong[:, :, 1:]
cBuy = np.logical_and(np.logical_not(cPrev), cNow)
cSell = np.logical_and(cPrev, np.logical_not(cNow))

numSym, popSize = cSell.shape[1], pop.shape[0]
totGains = np.ones((pop.shape[0], numSym))
opens = np.ones((popSize, numSym))
handicap = 0.003
for t in range(cSell.shape[2]):
    stack = np.stack([hist[:, t, 3] for _ in range(popSize)])
    opens = cBuy[:, :, t] * stack + np.logical_not(cBuy[:, :, t]) * opens
    gains = stack / opens * (1 - handicap)
    totGains = cSell[:, :, t] * gains + np.logical_not(cSell[:, :, t]) * np.ones((popSize, numSym))
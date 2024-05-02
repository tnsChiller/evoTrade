import numpy as np
import Utilities as util
import IndicatorsVectorized as ind
import Aux
import time

hist = Aux.GetHist()
metrics = util.GetMetrics(hist)
popSize = 100
pop = util.StartMetricPopulation(metrics, popSize)

t0 = time.perf_counter()


popSize = pop.shape[0]
w, thr = pop[:, :metrics.shape[0]], pop[:, metrics.shape[0]:]
w = w.reshape((popSize, metrics.shape[0],1 ,1))
scr = (w * metrics).sum(axis = 1)
c0 = np.zeros((popSize, metrics.shape[1], metrics.shape[2]), bool)
c1 = np.zeros((popSize, metrics.shape[1], metrics.shape[2]), bool)
c0 = scr > thr[:, 0].reshape(thr.shape[0], 1, 1)
c1 = scr < thr[:, 1].reshape(thr.shape[0], 1, 1)

cBuy = np.logical_and(c0[:, :, 1:], np.logical_not(c0[:, :, :-1]))
cSell = np.logical_and(c0[:, :, 1:], np.logical_not(c0[:, :, :-1]))


Aux.ReportTime([t0, time.perf_counter()])
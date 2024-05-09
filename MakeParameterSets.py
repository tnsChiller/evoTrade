import numpy as np
import Utilities as util
import Ext

hist0 = Ext.GetHist()
vSplit = 0.2
hist = hist0[:, :int(hist0.shape[1] * (1 - vSplit)), :]
histV = hist0[:, int(hist0.shape[1] * (1 - vSplit)):, :]
metrics = util.GetMetrics(hist)
# popSize, pieceSize, gens = 5000, 200, 100
popSize,  gens = 800, 20
pop = util.StartMetricPopulation(metrics, popSize)
(pop, gains) = util.EvolvePopulation(hist, gens, pop)
(pop, gainsV) = util.EvolvePopulation(histV, 0, pop)

res = np.concatenate((gainsV.reshape(popSize, 1),
                      gains.reshape(popSize, 1),
                      pop), axis = 1)
res = res[res[:, 0].argsort()]

selection = []
for idx in range(popSize - 20, popSize):
    if res[idx, 1] > 2 and res[idx, 0] > 2:
        selection.append([res[idx, 1], res[idx, 0], res[idx, 2:]])
        util.SaveParameterSet(res[idx, 2:], "-", res[idx, 1], res[idx, 0])
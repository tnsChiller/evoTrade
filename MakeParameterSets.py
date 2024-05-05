import numpy as np
import Utilities as util
import Ext

hist0 = Ext.GetHist()
vSplit = 0.2
hist = hist0[:, :int(hist0.shape[1] * (1 - vSplit)), :]
histV = hist0[:, int(hist0.shape[1] * (1 - vSplit)):, :]
metrics = util.GetMetrics(hist)
# popSize, pieceSize, gens = 5000, 200, 20
popSize,  gens = 800, 150
pop = util.StartMetricPopulation(metrics, popSize)
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

(pop, gainsV) = util.EvolvePopulation(histV, 0, pop)

res = np.concatenate((gainsV.reshape(popSize, 1),
                      gains.reshape(popSize, 1),
                      pop), axis = 1)
res = res[-res[:, 0].argsort()]

selection = []
for i in range(5):
    idx = int(popSize * 0.2) - 1 - i
    if gains[idx] > 2 * gainsV[idx]:
        selection.append([gainsV[idx], gains[idx], pop[idx]])
        util.SaveParameterSet(pop[idx], "-", gains[idx], gainsV[idx])
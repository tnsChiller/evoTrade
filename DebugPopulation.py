import Utilities as util
import numpy as np
import Ext

pSets = util.LoadFile("pSets")
pop = np.array([pSets[pSet]["pSet"] for pSet in pSets])
hist = Ext.GetHist()
gens = 1
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

idx = np.where(gains == gains.max())[0][0]
dbPop = np.array([pop[idx]])
metrics = util.GetMetrics(hist)
(cBuy, cSell) = util.GetConds(metrics, dbPop)

popSize = cBuy.shape[0]
numSym = cSell.shape[1]
totGains = np.ones((popSize, numSym))
opens = np.ones((popSize, numSym))
handicap = 0.003
b, c = 50, 1
a = 1 / np.log(b + 1)
for t in range(cSell.shape[2]):
    stack = np.stack([hist[:, t, 3] for _ in range(popSize)])
    opens = cBuy[:, :, t] * stack + np.logical_not(cBuy[:, :, t]) * opens
    gains = stack / opens * (1 - handicap)
    gains = a * np.log(b * gains + c)
    totGains *= cSell[:, :, t] * gains + np.logical_not(cSell[:, :, t]) * np.ones((popSize, numSym))
    
print(f"{totGains.prod(axis = 1)}")

a = 1
for i in totGains[0]:
    a *= i
    print(a)
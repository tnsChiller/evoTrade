import Utilities as util
import Ext

loadPop = True
hist = Ext.GetHist()
metrics = util.GetMetrics(hist)
popSize,  gens = 1000, 1000
if loadPop:
    [pop, gains] = util.LoadFile("lastPop")
    
else:
    pop = util.StartMetricPopulation(metrics, popSize)
    
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

modelsToAdd = 10
splitIdx = int(popSize * 0.2)
for i in range(splitIdx - modelsToAdd, splitIdx):
    util.SaveParameterSet(pop[i], "-", gains[i], -1, "evo_mkI")
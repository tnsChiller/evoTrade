import numpy as np
import Utilities as util
import pickle

with open("lastHist.pickle", "rb") as f:
    hist = pickle.load(f)
    
loadPop = True
metrics = util.GetMetrics(hist)
popSize,  gens = 1000, 2000
if loadPop:
    [pop, gains] = util.LoadFile("lastPop")
    
else:
    pop = util.StartMetricPopulation(metrics, popSize)
     
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

modelsToAdd = 5
splitIdx = int(popSize * 0.2)
for i in range(splitIdx - modelsToAdd, splitIdx):
    util.SaveParameterSet(pop[i], "-", gains[i], -1, "evo_mkI")
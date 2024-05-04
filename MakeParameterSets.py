import numpy as np
import Utilities as util
import uuid
import Ext

hist = Ext.GetHist()
metrics = util.GetMetrics(hist)
popSize, gens = 200, 10
pop = util.StartMetricPopulation(metrics, popSize)
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

for i in range(20):
    if i == 0:
        idx = int(popSize * 0.2) - 1
    else:
        idx = np.random.randint(0, popSize * 0.2)
        
    util.SaveParameterSet(pop[idx], gains[idx], str(uuid.uuid4()))
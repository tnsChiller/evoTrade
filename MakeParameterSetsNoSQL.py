import numpy as np
import Utilities as util
import pickle
import uuid

with open("lastHist.pickle", "rb") as f:
    hist = pickle.load(f)
    
metrics = util.GetMetrics(hist)
popSize, gens = 1000, 1000
pop = util.StartMetricPopulation(metrics, popSize)
(pop, gains) = util.EvolvePopulation(hist, gens, pop)

randomPicks = 5
pSets = [pop[np.random.randint(0, popSize * 0.2)] for _ in range(randomPicks)]
for pSet in pSets:
    util.SaveParameterSet(pSet, str(uuid.uuid4()))
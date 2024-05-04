import numpy as np
import pandas as pd
import Utilities as util
import IndicatorsVectorized as ind
import Aux
import time

hist = Aux.GetHist()
metrics = util.GetMetrics(hist)
popSize = 1000
pop = util.StartMetricPopulation(metrics, popSize)

gens = 60
for gen in range(gens):
    (cBuy, cSell) = util.GetConds(metrics, pop)
    gains = util.GetGainsMetric(hist, cBuy, cSell, pop)
    pop = util.NextGeneration(gains, pop)
    print(f"Gen = {gen}, max scr = {round(gains.max(),4)}")
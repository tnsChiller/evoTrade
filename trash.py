import numpy as np
import Utilities as util
import IndicatorsVectorized as ind
import time

hist = util.GetHist()
metrics = util.GetMetrics(hist)
popSize = 100
pop = util.StartMetricPopulation(metrics, popSize)

t0 = time.perf_counter()
gains = util.GetGainsMetric(hist, metrics, pop)
print(f"t = {round(time.perf_counter() - t0, 4)}")
print(f"t / pop = {round((time.perf_counter() - t0) / popSize, 4)}")
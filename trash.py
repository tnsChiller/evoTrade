import numpy as np
import Utilities as util
import time

hist = util.GetHist()
inds = util.Get_Inds(hist)
popSize = 1000
pop = util.StartPopulation(inds, popSize)

t0 = time.perf_counter()
totGains = util.GetGainsVect(hist, inds, pop)
t1 = time.perf_counter()
print(f"t = {round(t1 - t0, 4)}")
print(f"t / pop = {round((t1 - t0) / popSize, 4)}")
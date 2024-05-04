import numpy as np
import pandas as pd
import Utilities as util
import IndicatorsVectorized as ind
import Aux
import time

hist = Aux.GetHist()
metrics = util.GetMetrics(hist)
popSize = 200
pop = util.StartMetricPopulation(metrics, popSize)

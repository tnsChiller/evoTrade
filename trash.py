import numpy as np
import pandas as pd
import Utilities as util
import IndicatorsVectorized as ind
import Ext
import time

hist = Ext.GetHist()
metrics = util.GetMetrics(hist)
popSize = 200
pop = util.StartMetricPopulation(metrics, popSize)

import numpy as np
import Utilities as util

hist = util.GetHist()
inds = util.Get_Inds(hist)
pop = util.StartPopulation(inds, 1)

thrs = pop[0]
######
c0, c1 = inds < thrs[0], inds > thrs[1]
cLong = np.logical_and(np.all(c0, axis = 0), np.all(c1, axis = 0))
cPrev, cNow = cLong[:, :-1], cLong[:, 1:]
cBuy = np.logical_and(np.logical_not(cPrev), cNow)
cSell = np.logical_and(cPrev, np.logical_not(cNow))

numSym = cSell.shape[0]
pos = np.zeros(numSym, bool)
totGains = np.ones(numSym, np.float32)
opens = np.ones(numSym, np.float32)
handicap = 0.002
for i in range(cSell.shape[1]):
    opens = cBuy[:, i] * hist[:, i, 3] + np.logical_not(cBuy[:, i]) * opens
    pos = cBuy[:, i] * np.ones(numSym, bool) + np.logical_not(cBuy[:, i]) * pos
    gains = hist[:, i, 3] / opens * handicap
    totGains *= cSell[:, i] * gains + np.logical_not(cSell[:, i]) * np.ones(numSym, bool)
    pos = cSell[:, i] * np.zeros(numSym, bool) + np.logical_not(cSell[:, i]) * pos
######
import Utilities as util
import Analysis as anal

pSets = util.ResetPSets()
# hist = Ext.GetHist()
hist = util.LoadFile("lastHist")
tStart = int(hist.shape[1] * 0.85)
l = 200

# obl = anal.GetObl(pSets, tStart, hist, l)
obl = util.LoadFile("orderBigList")
summary = anal.GetSummary(obl)
gains = anal.GetModGains(summary)

for gain in gains:
    print(f"type: {pSets[gain]['type']}, gain: {gains[gain]['rating']}, id: {gain[:6]}")
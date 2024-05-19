import Utilities as util
import pandas as pd
import uuid

def GetObl(pSets, tStart, hist, l):
    obl = {}
    for t in range(tStart, hist.shape[1] + 1):
        subHist = hist[:, t - l:t]
        metrics = util.GetMetrics(subHist)
        moves = util.GetMoves(pSets, metrics, verbose = False)
        orderList = util.CreateOrderListMatrix(moves, subHist, pSets, t - 1)
        for order in orderList:
            obl[str(uuid.uuid4())] = order
            
        if (t - tStart) % 25 == 0:
            print(f"Processing history: {t - tStart + 1} / {hist.shape[1] - tStart}")
            
    return obl

def GetSummary(obl):
    oblDf = pd.DataFrame(obl).T
    summary = {}
    models = oblDf["model"].unique()
    for model in models:
        modSum = {}
        modDf = oblDf[oblDf["model"] == model]
        syms = modDf["sym"].unique()
        # print(f"model: {model}")
        for sym in syms:
            symDf = modDf[modDf["sym"] == sym]
            modSum[sym] = symDf
            sides = []
            for i in range(len(symDf["side"])):
                sides.append(symDf["side"][symDf.index[i]])
            # print(f"{sym}: {sides}")
        summary[model] = modSum
        
    return summary

def GetModGains(summary):
    gains = {}
    for model in summary:
        modGains = {}
        rating = 1
        for sym in summary[model]:
            symGains = summary[model][sym]
            gain = 1
            if len(symGains) > 1:
                if len(symGains) % 2 == 1:
                    symGains = symGains.drop(symGains.index[-1])
                    
                idx = symGains.index
                if symGains["side"][idx[0]] == "sell":
                    print(f"Warining: model {model} sold {sym} before buying.")
        
                for i in idx:
                    if symGains["side"][i] == "buy":
                        gain /= symGains["price"][i]
                        
                    else:
                        gain *= symGains["price"][i]
                
            modGains[sym] = gain
            rating *= gain
        gains[model] = {"rating": round(rating, 4), "gains": modGains}
        
    return gains
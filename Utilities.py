import gc
import pandas as pd
import numpy as np
import IndicatorsVectorized as ind
from sqlalchemy import create_engine
gc.enable()

def GetGainsVect(hist, inds, pop):
    c0, c1 = inds < pop[:, 0], pop[:, 1]
    cLong = np.logical_and(np.all(c0, axis = 1), np.all(c1, axis = 1))
    cPrev, cNow = cLong[:, :, :-1], cLong[:, :, 1:]
    cBuy = np.logical_and(np.logical_not(cPrev), cNow)
    cSell = np.logical_and(cPrev, np.logical_not(cNow))
    
    numSym, popSize = cSell.shape[1], pop.shape[0]
    totGains = np.ones((pop.shape[0], numSym))
    opens = np.ones((popSize, numSym))
    handicap = 0.003
    for t in range(cSell.shape[2]):
        stack = np.stack([hist[:, t, 3] for _ in range(popSize)])
        opens = cBuy[:, :, t] * stack + np.logical_not(cBuy[:, :, t]) * opens
        gains = stack / opens * (1 - handicap)
        totGains = cSell[:, :, t] * gains + np.logical_not(cSell[:, :, t]) * np.ones((popSize, numSym))
    return totGains

def GetGains(hist, inds, thrs):
    c0, c1 = inds < thrs[0], inds > thrs[1]
    cLong = np.logical_and(np.all(c0, axis = 0), np.all(c1, axis = 0))
    cPrev, cNow = cLong[:, :-1], cLong[:, 1:]
    cBuy = np.logical_and(np.logical_not(cPrev), cNow)
    cSell = np.logical_and(cPrev, np.logical_not(cNow))
    
    numSym = cSell.shape[0]
    totGains = np.ones(numSym, np.float32)
    opens = np.ones(numSym, np.float32)
    handicap = 0.002
    for i in range(cSell.shape[1]):
        opens = cBuy[:, i] * hist[:, i, 3] + np.logical_not(cBuy[:, i]) * opens
        gains = hist[:, i, 3] / opens * (1 - handicap)
        totGains *= cSell[:, i] * gains + np.logical_not(cSell[:, i]) * np.ones(numSym, bool)
    
    return totGains

def GetHist():
    engine = create_engine('postgresql+psycopg2://newuser:password@localhost:5432/postgres')
    
    play_list = ['AAPL','MSFT','GOOG','AMZN','NVDA','TSLA','META','LLY',
                 'V','XOM','UNH','WMT','JPM','MA','JNJ','PG','AVGO','ORCL','HD',
                 'CVX','MRK','ABBV','ADBE','KO','COST','PEP','CSCO','BAC','CRM',
                 'MCD','TMO','NFLX','PFE','CMCSA','DHR','ABT','AMD','TMUS','INTC',
                 'INTU','WFC','TXN','NKE','DIS','COP','CAT','PM','MS','VZ','AMGN',
                 'UPS','NEE','IBM','LOW','UNP','BA','BMY','SPGI','AMAT','HON',
                 'NOW','GE','RTX','QCOM','AXP','DE','PLD','SYK','SBUX',
                 'SCHW','GS','LMT','ELV','ISRG','TJX','BLK','T','ADP','UBER',
                 'MMC','MDLZ','GILD','ABNB','REGN','LRCX','VRTX','ADI','ZTS',
                 'SLB','CVS','AMT','CI','BX','PGR','BSX','MO','C','BDX']
    
    sql = 'SELECT * FROM df_m60;'
    df = pd.read_sql(sql, con=engine)
    hist = np.stack([pd.concat([df[f'{symbol}_Open'],
                     df[f'{symbol}_High'],
                     df[f'{symbol}_Low'],
                     df[f'{symbol}_Close'],
                     df[f'{symbol}_Volume']],axis=1).to_numpy(np.float32) for symbol in play_list])
    
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if np.any(np.isnan(hist[i, j])):
                hist[i, j] = hist[i, j-1]
                
    return hist

def GetMetrics(hist):
    c0 = hist[:, :, 4] == 0
    hist[:, :, 4] = c0 * np.ones(hist.shape[1]) + np.logical_not(c0) * hist[:, :, 4]
    
    base = {"close": hist[:, :, 3], "vol": hist[:, :, 4]}
    inds = {}
    ns = [6, 30, 120]
    l = hist.shape[1] - max(ns)
    for typ in ["close", "vol"]:
        tmp0 = {}
        for n in ns:
            tmp1 = {}
            tmp1["ma"], tmp1["std"] = ind.MovingAverageStd(base[typ], n)
            tmp1["rsi"] = ind.Rsi(hist, n)
            
            start = tmp1["ma"].shape[1] - l
            tmp1["ma"], tmp1["std"] = tmp1["ma"][:, start + 1:], tmp1["std"][:, start + 1:]
            tmp1["rsi"] = tmp1["rsi"][:, start:]
            
            tmp1["bbhi"], tmp1["bblo"] = tmp1["ma"] + tmp1["std"], tmp1["ma"] - tmp1["std"]
            tmp0[str(n)] = tmp1
        tmp0["1"] = {"ma": base[typ][:, hist.shape[1] - l + 1:]}
        inds[typ] = tmp0
        
    alpha = ((1, 6), (1, 30), (1, 120), (6, 30), (6, 120), (30, 120),
             (6, 6), (30, 30), (120, 120))
    beta = ((1, 6), (1, 30), (1, 120), (6, 30), (6, 120), (30, 120))
    gamma = ((1, 6), (1, 30), (1, 120), (6, 30), (6, 120), (30, 120))
    metrics = []
    
    for i in alpha:
        metrics.append(inds["close"][str(i[1])]["bblo"] / inds["close"][str(i[0])]["ma"])
        metrics.append(inds["close"][str(i[0])]["ma"] / inds["close"][str(i[1])]["bbhi"])
        
    for i in beta:
        metrics.append(inds["close"][str(i[0])]["ma"] / inds["close"][str(i[1])]["ma"])
        
    for i in gamma:
        metrics.append(inds["vol"][str(i[0])]["ma"] / inds["vol"][str(i[1])]["ma"])
        
    for i in ns:
        metrics.append(inds["close"][str(i)]["rsi"])
        
    return np.array(metrics)

def GetInds(hist):
    all_inds = []
    close = np.expand_dims(hist[:, -1, 3], axis = 1)
    vol5, _ = ind.MovingAverageStd(hist[:, :, 4], 5)
    vol5 = np.expand_dims(vol5[:, -1], axis = 1)
    for n in [5, 10, 20, 50]:
        rs = ind.Rsi(hist, n)
        bb = ind.BollingerBands(hist, n)
        hi = bb[0] / close
        lo = bb[1] / close
        so = ind.StochasticOscillator(hist, n)
        ma, st = ind.MovingAverageStd(hist[:, :, 3], n)
        ma, st = ma / close, st / close
        vm, vs = ind.MovingAverageStd(hist[:, :, 4], n)
        vm, vs = vs / vol5, st / vol5
        # Normalise
        all_inds.append(rs)
        all_inds.append(hi)
        all_inds.append(lo)
        all_inds.append(so)
        all_inds.append(ma)
        all_inds.append(st)
        all_inds.append(vm)
        all_inds.append(vs)
    
        all_inds.append(rs[:, 1:] - rs[:, :-1])
        all_inds.append(hi[:, 1:] - hi[:, :-1])
        all_inds.append(lo[:, 1:] - lo[:, :-1])
        all_inds.append(so[:, 1:] - so[:, :-1])
        all_inds.append(ma[:, 1:] - ma[:, :-1])
        all_inds.append(st[:, 1:] - st[:, :-1])
        all_inds.append(vm[:, 1:] - vm[:, :-1])
        all_inds.append(vs[:, 1:] - vs[:, :-1])
    
    min_length = min(_.shape[1] for _ in all_inds)
    inds_array = np.stack([_[:,-min_length:] for _ in all_inds])
    
    return inds_array

def GetRandomThresholdSet(inds):
    maxs = inds.max(axis = 2).max(axis = 1)
    mins = inds.min(axis = 2).min(axis = 1)
    thrs = np.random.rand(len(maxs)) * (maxs - mins) + mins
    thrs = thrs.reshape(thrs.shape[0], 1, 1)
    
    return thrs

def StartPopulation(inds, size):
    popList = [[GetRandomThresholdSet(inds) for i in range(2)] for k in range(size)]
    
    return np.array(popList)

def StartMetricPopulation(metrics, size):
    popList = [np.random.rand(metrics.shape[0] + 2) - 0.5 * 2 for _ in range(size)]
    
    return np.array(popList)
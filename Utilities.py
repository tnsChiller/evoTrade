import gc
import time
import uuid
import pickle
import datetime
import numpy as np
import IndicatorsVectorized as ind
from Constants import playList, cfg
gc.enable()

def RunSummary():
    models = {}
    orderBigList = LoadFile("orderBigList")
    for order in orderBigList:
        models[order["model"]] = 1

def WaitCycle():
    time.sleep(cfg["step"])

def GetGoTime(mnt, sec):
    minute = datetime.now().minute
    second = datetime.now().second
    c0 = minute == mnt
    c1 = second > sec
    
    return c0 and c1

def GetLiveMetrics(yfd):
    keys = list(yfd['Close'].keys())
    hist = np.stack((yfd['Open'], yfd['High'], yfd['Low'], yfd['Close'], yfd['Volume'])).T
    hist = hist[:, :hist.shape[1] - 1]
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if np.any(np.isnan(hist[i, j])):
                hist[i, j] = hist[i, j-1]
                
    return (GetMetrics(hist), keys)

def GetMoves(pSets, metrics, keys):
    moves = {}
    for pSet in pSets:
        if pSet["active"]:
            pop = np.array([pSets[pSet]["pSet"]])
            (cBuy, cSell) = GetConds(metrics, pop)
            moves[pSet] = {}
            for i in range(len(keys)):
                if cBuy[0, i, -1] or cSell[0, i, -1]:
                    print(pSet)
                    print(keys[i])
                moves[pSet][keys[i]] = {"buy": cBuy[0, i, -1], "sell": cSell[0, i, -1]}
    
    return moves
    
def CreateOrderList(moves, yfd):
    orderBigList = LoadFile("orderBigList")
    orderList = []
    for model in moves:
        for sym in moves[model]:
            if moves[model][sym]["buy"] and not moves[model]["status"]:
                order = {"model": model,
                         "sym": sym,
                         "side": "buy",
                         "qty": cfg["orderSize"] / yfd['Close'][sym][-1],
                         "price": yfd['Close'][sym][-1],
                         "time": datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S"),
                         "unid": uuid.uuid4()}
                orderList.append(order)
                moves[model]["status"] = True
                orderBigList[order["unid"]] = order
                
            if moves[model][sym]["sell"] and moves[model]["status"]:
                order = {"model": model,
                         "sym": sym,
                         "side": "sell",
                         "qty": cfg["orderSize"] / yfd['Close'][sym][-1],
                         "price": yfd['Close'][sym][-1],
                         "time": datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S"),
                         "unid": uuid.uuid4()}
                orderList.append(order)
                moves[model]["status"] = False
                orderBigList[order["unid"]] = order
                
    SaveFile("orderBigList")
                
    return orderList

def EvolveInPieces(hist, gens, pop, pieceSize):
    metrics = GetMetrics(hist)
    popSize = pop.shape[0]
    for gen in range(gens + 1):
        gains = np.ones(popSize)
        for piece in range(popSize // pieceSize):
            subPop = pop[piece * pieceSize:(piece + 1) * pieceSize]
            (subCBuy, subCSell) = GetConds(metrics, subPop)
            subGains = GetGainsMetric(hist, subCBuy, subCSell)
            gains[piece * pieceSize:(piece + 1) * pieceSize] = subGains
        
        if gen != gens:
            pop = NextGeneration(gains, pop)
        print(f"Gen = {gen}, mean scr = {round(gains[:int(pop.shape[0] * 0.4)].mean(),4)}")

def EvolvePopulation(hist, gens, pop):
    metrics = GetMetrics(hist)
    for gen in range(gens + 1):
        (cBuy, cSell) = GetConds(metrics, pop)
        gains = GetGainsMetric(hist, cBuy, cSell)
        if gen != gens:
            pop = NextGeneration(gains, pop)
        print(f"Gen = {gen}, mean scr = {round(gains[:int(pop.shape[0] * 0.4)].mean(),4)}")
    
    return (pop, gains)

def NextGeneration(scr, pop):
    popSize, metnum = pop.shape[0], pop.shape[1]
    metricScr = np.concatenate((scr.reshape((len(scr),1)), pop), axis = 1)
    metricScr = metricScr[metricScr[:, 0].argsort()]
    elite = metricScr[-int(metricScr.shape[0] * 0.2):, 1:]
    new = np.random.rand(int(popSize * 0.4), metnum)
    
    mutRatio, mutRange = 0.00, 2
    child = np.zeros_like(new)
    for i in range(child.shape[0]):
        for j in range(child.shape[1]):
            gene = elite[np.random.randint(0, elite.shape[0]), j]
            if np.random.rand() < mutRatio:
                gene *= np.random.rand() * mutRange
            child[i, j] = gene
    
    return np.concatenate((elite, new, child))

def GetFastestPopSize(hist):
    gc.enable()
    metrics = GetMetrics(hist)
    popSizes = np.arange(200, 5001, 200)
    for popSize in popSizes:
        t0 = time.perf_counter()
        pop = StartMetricPopulation(metrics, popSize)
        (cBuy, cSell) = GetConds(metrics, pop)
        gains = GetGainsMetric(hist, cBuy, cSell)
        pop = NextGeneration(gains, pop)
        t1 = time.perf_counter()
        print(f"popSize = {popSize}, t / popSize = {np.round((t1 - t0) / popSize, 3)}")

def GetConds(metrics, pop):
    w, thr = pop[:, :metrics.shape[0]], pop[:, metrics.shape[0]:]
       
    scr = np.matmul(w, metrics.swapaxes(0, 1)).swapaxes(0, 1)
    c0 = scr > thr[:, 0].reshape(thr.shape[0], 1, 1)
    c1 = scr < thr[:, 1].reshape(thr.shape[0], 1, 1)
    cBuy = np.logical_and(c0[:, :, 1:], np.logical_not(c0[:, :, :-1]))
    cSell = np.logical_and(c1[:, :, 1:], np.logical_not(c1[:, :, :-1]))
    
    buyLast = np.zeros((cBuy.shape[0], cBuy.shape[1]), bool)
    for t in range(cSell.shape[2]):
        cBuy[:, :, t] = np.logical_not(buyLast) * cBuy[:, :, t]
        cSell[:, :, t] = buyLast * cSell[:, :, t]
        
        buyLast = cBuy[:, :, t] + np.logical_not(cBuy[:, :, t]) * buyLast
        buyLast = np.logical_not(cSell[:, :, t]) * buyLast
        
    return (cBuy, cSell)

def GetGainsMetric(hist, cBuy, cSell):
    popSize = cBuy.shape[0]
    numSym = cSell.shape[1]
    totGains = np.ones((popSize, numSym))
    opens = np.ones((popSize, numSym))
    handicap = 0.003
    b, c = 50, 1
    a = 1 / np.log(b + 1)
    for t in range(cSell.shape[2]):
        stack = np.stack([hist[:, t, 3] for _ in range(popSize)])
        opens = cBuy[:, :, t] * stack + np.logical_not(cBuy[:, :, t]) * opens
        gains = stack / opens * (1 - handicap)
        gains = a * np.log(b * gains + c)
        totGains *= cSell[:, :, t] * gains + np.logical_not(cSell[:, :, t]) * np.ones((popSize, numSym))
    
    return totGains.prod(axis = 1)

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

def StartMetricPopulation(metrics, size):
    popList = [(np.random.rand(metrics.shape[0] + 2) - 0.5) * 4 for _ in range(size)]
    
    return np.array(popList, np.float32)

def SaveParameterSet(pSet, name, scr, scrV):
    status = {}
    for sym in playList:
        status[sym] = False
    with open("pSets.pickle", "rb") as f:
        pSets = pickle.load(f)
    
    pSetEntry = {"name": name,
                 "scr": scr,
                 "scrV": scrV,
                 "status": status,
                 "active": True,
                 "type": "evo_mkI",
                 "pSet": pSet,
                 "date": datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")}
    pSets[str(uuid.uuid4())] = pSetEntry
    with open("pSets.pickle", "wb") as f:
        pickle.dump(pSets, f, pickle.HIGHEST_PROTOCOL)
        
def LoadParameterSets():
    with open("pSets.pickle", "rb") as f:
        pSets = pickle.load(f)
        
    return pSets

def SaveFile(obj, name):
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def LoadFile(name):
    with open(f"{name}.pickle", "rb") as f:
        obj = pickle.load(f)
        
    return obj
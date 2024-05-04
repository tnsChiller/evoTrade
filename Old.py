def GetGains(hist, inds, pop):
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
        totGains = cSell[:, :, t] * gains + np.logical_not(cSell[:, :, t]) * totGains
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
    
    return np.array(popList, np.float32)
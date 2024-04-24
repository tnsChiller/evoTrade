import numpy as np
hist, n, k = np.random.rand(1000,5), 15, 2

def MovingAverageStd(array, n):
    stack = np.stack([array[i : len(array)-n+i+1] for i in range(n)])
    ma = stack.mean(axis = 0)
    std = stack.std(axis = 0)
    return ma, std

def Rsi(hist, n):
    hist0 = hist[: -1]
    hist1 = hist[1 :]
    
    u = hist1[:, 3] - hist0[:, 3]    
    u = u * (u > 0)
    uSmooth, _ = MovingAverageStd(u, n)
    
    d = hist0[:, 3] - hist1[:, 3]
    d = d * (d > 0)
    dSmooth, _ = MovingAverageStd(d, n)
    return 1 - 1 / (1 + np.divide(uSmooth, dSmooth, out = 100 * np.ones_like(uSmooth), where=dSmooth != 0))

def BollingerBands(hist, n):
    ma, std = MovingAverageStd(hist[:,3], n)
    out = np.stack([ma + std, ma - std]).T
    return out

def StochasticOscillator(hist, n):
    maxs = np.stack([hist[i:len(hist)-n+i+1,1] for i in range(n)]).max(axis = 0)
    mins = np.stack([hist[i:len(hist)-n+i+1,2] for i in range(n)]).min(axis = 0)
    out = (hist[n-1:,3] - mins) / (maxs - mins)
    return out
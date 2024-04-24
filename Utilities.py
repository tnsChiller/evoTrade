import gc
import pandas as pd
import numpy as np
import IndicatorsVectorized as ind
from sqlalchemy import create_engine
gc.enable()

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

def Get_Inds(hist):
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
    
    return [[GetRandomThresholdSet(inds) for i in range(2)] for k in range(size)]
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from Constants import playList

def GetHist():
    engine = create_engine('postgresql+psycopg2://newuser:password@localhost:5432/postgres')
    
    sql = 'SELECT * FROM df_m60;'
    df = pd.read_sql(sql, con=engine)
    hist = np.stack([pd.concat([df[f'{symbol}_Open'],
                     df[f'{symbol}_High'],
                     df[f'{symbol}_Low'],
                     df[f'{symbol}_Close'],
                     df[f'{symbol}_Volume']],axis=1).to_numpy(np.float32) for symbol in playList])
    
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            if np.any(np.isnan(hist[i, j])):
                hist[i, j] = hist[i, j-1]
                
    return hist

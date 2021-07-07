import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataPretreatment(path):
    data=pd.read_csv(path)          # as usual,we need transfrom rawdata into ndarray
    data=np.array(data)

    score=data[:,0:-1]      # score correspond to the input data(x)
    y=data[:,-1]           # y correspond to if the student is qualified

    [row,col]=score.shape
    bias=np.ones(row)
    score=np.insert(score,0,bias,axis=1)

    return score,y 

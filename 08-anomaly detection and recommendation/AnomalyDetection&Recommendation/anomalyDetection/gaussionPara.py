import numpy as np

def gaussionPara(x):
    mean=np.mean(x,axis=0)
    var=np.var(x,axis=0)
    return mean,var
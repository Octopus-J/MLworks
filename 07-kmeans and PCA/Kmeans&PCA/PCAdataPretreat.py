import numpy as np

def dataTreat(x):
    x=x-np.mean(x,axis=0)        # mean value normalization, features has similar scales, so no need to feature scaling

    sigma=((x.T)@x)/len(x)       
    U,S,V=np.linalg.svd(sigma)   # SVD
    return x,U,S,V
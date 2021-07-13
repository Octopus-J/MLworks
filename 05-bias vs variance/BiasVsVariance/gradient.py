import numpy as np
from hypothesis import hyp

def grd(theta,x,y,lamda):
    m=x.shape[0]
    theta=theta.reshape(x.shape[1],1)

    para1=(x.T@(hyp(theta,x)-y))
    para2=theta[1:]
    para2=(np.insert(para2,0,0,axis=0))*lamda    # don't punish theta0

    return (para1+para2)/m
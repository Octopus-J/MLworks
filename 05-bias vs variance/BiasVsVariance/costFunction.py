import numpy as np
from hypothesis import hyp

def cost(theta,x,y,lamda):
    m=x.shape[0]
    para1=np.sum(np.power((hyp(theta,x)-y),2))
    para2=np.sum(np.power(theta[1:],2))                               # don't punish theta0

    return (para1+lamda*para2)/(2*m)
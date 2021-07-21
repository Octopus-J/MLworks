import numpy as np
from groupParams import groupParams
from groupParams import ungroupParams

def costFunction(paras,r,y,n_movies,n_users,n_features,lamda):
    x,theta=ungroupParams(paras,n_movies,n_users,n_features)

    err=0.5*(((x@theta.T-y)*r)**2).sum()            # *r is to remove those movies that were not actually rated
    reg1=(lamda/2)*np.sum(x**2)
    reg2=(lamda/2)*np.sum(theta**2)
    return err+reg1+reg2
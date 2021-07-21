import numpy as np
from groupParams import groupParams
from groupParams import ungroupParams

def grd(paras,r,y,n_movies,n_users,n_features,lamda):
    x,theta=ungroupParams(paras,n_movies,n_users,n_features)
    x_grd=((x@theta.T-y)*r)@theta+lamda*x
    theta_grd=((x@theta.T-y)*r).T@x+lamda*theta
    return groupParams(x_grd,theta_grd)
import numpy as np
from scipy.optimize import minimize
from costFunction import cost
from gradient import grd

def trainModel(x,y,lamda):
    theta=np.random.randint(100,200,size=(x.shape[1],1))                   # theta 2*1, incldue theta0, theta1
    result=minimize(fun=cost,
                    x0=theta,
                    args=(x,y,lamda),
                    method='TNC',
                    jac=grd)
    final_theta=result.x
    #print(final_theta)

    return final_theta
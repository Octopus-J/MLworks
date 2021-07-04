import numpy as np
import sigmoidFunction as sig

def costFunction(theta,x,y,flag=0,lamda=0):    # flag is used to control process,and lamda, the regularization parameter
    hyp=sig.sigmoidFunction(theta,x)
    epsilon=1.0e-200                           # epsilon is a very small constan, the reason to use it is to avoid the np.log(0),this problem troubled me a lot
    if flag==0:
        cost1=-(y.T)*(np.log(hyp+epsilon))
        cost2=-(1-y)*(np.log(1-hyp+epsilon))
        cost=np.sum(cost1+cost2)/len(y)
    else:
        _theta=theta[1:]                       
        _theta=np.insert(_theta,0,0)           # in python, if a parameter is variable,then in the transfer behavior,itself could be changed
        cost1=-(y.T)*(np.log(hyp+epsilon))
        cost2=-(1-y)*(np.log(1-hyp+epsilon))
        cost=np.sum(cost1+cost2)/(2*len(y))+(lamda/(2*len(y)))*np.sum(np.power(_theta,2))

    return cost
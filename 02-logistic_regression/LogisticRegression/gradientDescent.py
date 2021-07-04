import numpy as np
import sigmoidFunction
import costFunction

def gradientDescent(theta,x,y,alpha,iters,flag=0,lamda=0):   
    m=len(y)
    cost=np.zeros(iters)
    if (flag==0):       # no regularization
        for i in range(0,iters):
            sig=sigmoidFunction.sigmoidFunction(theta,x)
            theta=theta-(alpha/m)*(x.T@(sig-y))
            cost[i]=costFunction.costFunction(theta,x,y)      # record the cost value in each iteration
    else:               # with regularization
        for i in range(0,iters):
            _theta=theta[1:]                                  # we didn't punish the theta0, so in the graident descent process, it's 0
            _theta=np.insert(_theta,0,0)                      # notice!,here we can't use _theta[0]=0,because in python if a=1,b=a, then both a and b is the pointer of 1
            sig=sigmoidFunction.sigmoidFunction(theta,x)
            para1=(alpha/m)*(x.T@(sig-y))
            para2=(alpha*lamda/m)*_theta
            theta=theta-para1-para2
            cost[i]=costFunction.costFunction(theta,x,y,flag,lamda)      # record the cost value in each iteration

    return cost,theta 
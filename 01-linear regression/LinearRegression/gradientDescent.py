import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import costFunction
import hypothesis

def gradientDescent(theta,x,y,alpha,iters):     #iters is the number of iterations
    i=0
    cost=np.zeros(iters)              # save the value of cost function in each iteration
    predictValue=hypothesis.hypothesis(theta,x)
    while i<iters:
        theta=theta-alpha*(np.dot((x.T),(predictValue-y)))/(len(y))     # gradient descent of theta between
        #print(theta)
        predictValue=hypothesis.hypothesis(theta,x)              # update the predictValue 
        Jtheta=costFunction.costFunction(predictValue,y)             # update the cost function
        cost[i]=Jtheta                       # record the cost function
        i+=1
    return theta,cost

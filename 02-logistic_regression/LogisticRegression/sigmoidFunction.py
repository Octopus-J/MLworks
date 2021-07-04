import numpy as np

def sigmoidFunction(theta,x):   # the hypothesis of logistic regression
    z=x@theta
    gtheta=1/(1+np.exp(-z))     # the sigmoid function
    return gtheta
import numpy as np
import sigmoidFunction 

def costFunction(theta,x,y,lamda):           # theta 401*1,x 5000*401,y 5000*1,notice that theta must be the first parameter because of the claim of minimize.
    m=len(y)
    epsilon=1.0e-200                         # epsilon is a very small value, used to avoid the log(0)
    sig=sigmoidFunction.sig(x@theta)         # sig=5000*1

    para1=(y.T)@np.log(sig+epsilon)          # 1*1 

    para2=((1-y).T)@np.log(1-sig+epsilon)    # 1*1 

    reg=((theta[1:]).T@(theta[1:]))*lamda    # 1*1

    return (para1+para2)/m+reg/(2*m)         
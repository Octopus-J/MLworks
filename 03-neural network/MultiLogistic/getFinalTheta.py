import numpy as np
from scipy.optimize import minimize
import costFunction
import gradient

def finalTheta(x,y,lamda,k):        # k is the number of labels, in this question,k is 10
    n=x.shape[1]                    # get the number of features (401),x 5000*401,y 5000*1,k 10

    theta=np.zeros((k,n))           # create a zero ndarray (10*401) to store the final theta, 10 kinds of model in total,each has 401 theta

    for i in range(1,k+1):
        tempTheta=np.random.randint(10,size=n)                 # tempTheta 1*401
        tempTheta=tempTheta.reshape((401,1))  # tempTheta 401*1

        _y=(y==i)
        _y=_y*1                                     # let the other parameters (except i) be 0

        result=minimize(fun=costFunction.costFunction,
                        x0=tempTheta,                   # 401*1
                        args=(x,_y,lamda),              # x 5000*401,y 5000*1,
                        method='TNC',
                        jac=gradient.grd)               # use the minimize function, grd returns 401*1

        print(result.fun,result.message)
        theta[i-1,:]=result.x

    return theta
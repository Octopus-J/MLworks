import numpy as np
from scipy.optimize import minimize
from costFunction import costFunction
from gradient import grd

def finalTheta(x,y,lamda,k):        # k is the number of labels, in this question,k is 10
    n=x.shape[1]                    # get the number of features (401),x 5000*401,y 5000*1,k 10

    theta=np.random.random((k,n))                            # create a zero ndarray (10*401) to store the final theta, 10 kinds of model in total,each has 401 theta

    for i in range(1,k+1):
        print('i=',i)
        tempTheta=np.random.random((401,1))                  # tempTheta 401*1

        singleY=(y==i)
        singleY=singleY*1                                    # let the other parameters (except i) be 0

        # singleY = np.array([1 if label ==i else 0 for label in y]).reshape(5000,1)    # another way to get singleY


        result=minimize(fun=costFunction,
                        x0=tempTheta,                        # 401*1
                        args=(x,singleY,lamda),              # x 5000*401,y 5000*1,
                        method='TNC',
                        jac=grd)                             # use the minimize function, grd returns 401*1
        print(result.fun,result.message)
        theta[i-1,:]=result.x

    return theta
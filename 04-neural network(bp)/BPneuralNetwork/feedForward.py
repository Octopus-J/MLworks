import numpy as np
from sigmoidFunction import sig
import thetaGroup

def FF(theta,x):
    theta1,theta2=thetaGroup.thetaUnGroup(theta)
                    #######  input layer  #######
    a1=x                           # a1 5000*401, the input value of input layer(also the output value)

                    #######  hidden layer #######
    z2=a1@theta1.T                 # z2 5000*25, the input value of hidden layer

    a2=sig(z2) 
    a2=np.insert(a2,0,1,axis=1)    # a2 5000*26, the output value of hidden layer,need insert bias theta0=1

                    #######  output layer  #######
    z3=a2@theta2.T                 # z3 5000*10, the input value of output layer

    a3=sig(z3)                     # a3 5000*10, the output value of output layer

    return a1,z2,a2,z3,a3
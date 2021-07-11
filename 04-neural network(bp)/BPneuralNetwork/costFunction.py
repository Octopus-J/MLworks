import numpy as np
import thetaGroup
from feedForward import FF

def cost(theta,x,y,lamda):
    m=len(y)                                                                  # y 5000*10
    theta1,theta2=thetaGroup.thetaUnGroup(theta)                              # theta1 25*401,theta2 10*26
    _,_,_,_,a3=FF(theta,x)                                                    # a1 5000*401,z2 5000*25，a2 5000*26，z3 5000*10,a3 5000*10
    epsilon=1e-200                                                            # avoid log(0)

    para1=-np.sum(y*np.log(a3+epsilon)+(1-y)*np.log(1-a3+epsilon))
    
    para2=np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2))   # don't punish the theta0

    return (para1/m)+(para2*lamda/(2*m))

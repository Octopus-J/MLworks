import numpy as np
import thetaGroup
from feedForward import FF
from sigmoidFunction import sig_grd

def gradient(theta,x,y,lamda):
    m=len(y)
    theta1,theta2=thetaGroup.thetaUnGroup(theta)    # theta1 25*401,theta2 10*26
    a1,z2,a2,z3,a3=FF(theta,x)                      # a1 5000*401,z2 5000*25，a2 5000*26，z3 5000*10,a3 5000*10

    d3=a3-y                                         # d3 5000*10
    d2=(d3@theta2[:,1:])*sig_grd(z2)                # d2 5000*25

    uppD2=(d3.T@a2)/m                               # D2 10*26
    uppD1=(d2.T@a1)/m                               # D1 25*401

    tempTheta1=theta1[:,1:]
    tempTheta2=theta2[:,1:]
    tempTheta1=np.insert(tempTheta1,0,0,axis=1)
    tempTheta2=np.insert(tempTheta2,0,0,axis=1)

    uppD1=uppD1+tempTheta1*lamda/m                  # delta dosen't include the theta0
    uppD2=uppD2+tempTheta2*lamda/m

    all_D=thetaGroup.thetaGroup(uppD1,uppD2)
    return all_D
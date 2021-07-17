import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PCAdataPretreat import dataTreat

data=sio.loadmat('../data/ex7data1.mat')
x=data['X']                               # x 50*2

x,U,S,V=dataTreat(x)                      # each ! column ! of U is a feature vector

# Reduce dimension of x
Ureduce=U[:,0]
Ureduce=Ureduce.reshape(2,1)              # reduce dimension use first feature vector
z=x@Ureduce

Ureduce2=U[:,1]
Ureduce2=Ureduce2.reshape(2,1)            # reduce dimension use second feature vector
z2=x@Ureduce2

# reduction x
x_approx1=z@Ureduce.T
x_approx2=z2@Ureduce2.T

plt.scatter(x[:,0],x[:,1],facecolor='none',edgecolor='b')   # plot raw data

plt.plot([0,U[0,0]],
        [0,U[1,0]],
        c='r',linewidth=3,
        label='First Principle Component')# plot first column vector, also the first feature vector

plt.plot([0,U[0,1]],
        [0,U[1,1]],
        c='k',linewidth=3,
        label='Second Principle Component')#plot Second column vector, also the second feature vector

plt.grid()                                # show grid
plt.axis('equal')                         # make x and y axises has equl increments
plt.legend()                              # show legend

plt.scatter(x_approx1[:,0],x_approx1[:,1],label='First Principle Component dimension reduction')
plt.scatter(x_approx2[:,0],x_approx2[:,1],label='Second Principle Component dimension reduction')
plt.legend()
plt.show()
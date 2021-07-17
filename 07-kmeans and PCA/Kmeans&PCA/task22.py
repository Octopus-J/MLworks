import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PCAdataPretreat import dataTreat
# use PCA to reduce the dimension of pictures, requires to reduce to 36 dimensions

data=sio.loadmat('../data/ex7faces.mat')
rawx=data['X']        # 5000*1024

x,U,S,V=dataTreat(rawx)

Ureduce=U[:,0:36]          # 1024*36

z=x@Ureduce                # 5000*36, compressed data

x_approx=z@Ureduce.T       # 5000*1024, recovered data

# plot top 100 raw pictures
row=10
col=10
fix,ax1=plt.subplots(row,col,figsize=(8,8))
for r in range(row):
    for c in range(col):
        ax1[r][c].imshow(rawx[r*row+c].reshape(32,32).T,cmap='Greys_r')
        ax1[r][c].set_xticks([])   # hide x label
        ax1[r][c].set_yticks([])   # hide y label

# plot top 100 compressed pictures
row=10
col=10
fix,ax2=plt.subplots(row,col,figsize=(8,8))
for r in range(row):
    for c in range(col):
        ax2[r][c].imshow(z[r*row+c].reshape(6,6).T,cmap='Greys_r')
        ax2[r][c].set_xticks([])   # hide x label
        ax2[r][c].set_yticks([])   # hide y label

# plot top 100 recovered pictures
row=10
col=10
fix,ax3=plt.subplots(row,col,figsize=(8,8))
for r in range(row):
    for c in range(col):
        ax3[r][c].imshow(x_approx[r*row+c].reshape(32,32).T,cmap='Greys_r')
        ax3[r][c].set_xticks([])   # hide x label
        ax3[r][c].set_yticks([])   # hide y label

plt.show()
import numpy as np
import scipy.io as sio
from kmeans import kmeans
import matplotlib.pyplot as plt

# In this task, learn to implement kmeans
data=sio.loadmat('../data/ex7data2.mat')
x=data['X']    # x 300*2
k=3            # k is the number of classes
iters=30       # number of iterations

centers,class_all,center_note=kmeans(x,k,iters)

print(center_note.shape)

plt.scatter(x[:,0],x[:,1],c=class_all,cmap='rainbow')           # plot raw data

for i in range(k):                                              # plot each movement trajectory
    plt.plot(center_note[i,0,:],center_note[i,1,:],'kx--')      # center_noter[m,n,iters],(m,n) is the centers' value,iters is the number of iterations

plt.scatter(centers[:,0],centers[:,1],s=30,marker='o',c='k')   # plot final centers 
plt.show()
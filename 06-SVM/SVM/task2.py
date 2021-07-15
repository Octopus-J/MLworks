import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

def plotraw():
    plt.scatter(x[:,0],x[:,1],c=y.flatten(),cmap='bwr')
    plt.xlabel('x1')
    plt.ylabel('x2')

def plotBoundary(model):
    xx,yy=np.meshgrid(np.linspace(-0.1,1.1,500),np.linspace(0.3,1.1,500))   # get grid coordinates
    z=model.predict(np.c_[xx.flatten(),yy.flatten()])                     # predict which type each point in the grid belongs to
    zz=z.reshape(xx.shape)
    plt.contour(xx,yy,zz,[1])       # plot the decision boundary

np.set_printoptions(threshold=np.inf)
data = sio.loadmat('../data/ex6data2.mat')
x=data['X']      # x 863*2
y=data['y']      # y 863*1

# C=1,gamma=1
fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('C=1,gamma=1')
svc1=svm.SVC(C=1,kernel='rbf',gamma=1)   # create a svm with linear kernel, C=1, C is the parameter of penalty
svc1.fit(x,y.flatten())                   # train the svm, svm package will add the theta0 and x0
print(svc1.score(x,y.flatten()))          # print the accurancy of svc1

plotBoundary(svc1)                        # plot decision boundary
plotraw()                                 # plot raw data

# C=1,gamma=50
fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('C=1,gamma=50')
svc1=svm.SVC(C=1,kernel='rbf',gamma=50)   # create a svm with linear kernel, C=1, C is the parameter of penalty
svc1.fit(x,y.flatten())                   # train the svm, svm package will add the theta0 and x0
print(svc1.score(x,y.flatten()))          # print the accurancy of svc1

plotBoundary(svc1)                        # plot decision boundary
plotraw()                                 # plot raw data

# C=1,gamma=1000
fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('C=1,gamma=1000')
svc1=svm.SVC(C=1,kernel='rbf',gamma=1000)   # create a svm with linear kernel, C=1, C is the parameter of penalty
svc1.fit(x,y.flatten())                   # train the svm, svm package will add the theta0 and x0
print(svc1.score(x,y.flatten()))          # print the accurancy of svc1

plotBoundary(svc1)                        # plot decision boundary
plotraw()                                 # plot raw data
plt.show()

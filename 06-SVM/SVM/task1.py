import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from plotFunction import plotraw
from plotFunction import plotBoundary

np.set_printoptions(threshold=np.inf)
data = sio.loadmat('../data/ex6data1.mat')
x=data['X']      # x 51*2
y=data['y']      # y 51*1

# C=1
fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('C=1')
svc1=svm.SVC(C=1,kernel='linear')   # create a svm with linear kernel, C=1, C is the parameter of penalty
svc1.fit(x,y.flatten())             # train the svm, svm package will add the theta0 and x0
print(svc1.score(x,y.flatten()))    # print the accurancy of svc1

plotBoundary(svc1)                  # plot decision boundary
plotraw(x,y)                           # plot raw data
plt.show()

# C=100
fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('C=100')
svc1=svm.SVC(C=100,kernel='linear') # create a svm with linear kernel, C=100, C is the parameter of penalty
svc1.fit(x,y.flatten())             # train the svm, svm package will add the theta0 and x0
print(svc1.score(x,y.flatten()))    # print the accurancy of svc1

plotBoundary(svc1)                  # plot decision boundary
plotraw(x,y)                           # plot raw data
plt.show()
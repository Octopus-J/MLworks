import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

def plotraw():
    plt.scatter(x[:,0],x[:,1],c=y.flatten(),cmap='bwr')
    plt.xlabel('x1')
    plt.ylabel('x2')

def plotBoundary(model):
    xx,yy=np.meshgrid(np.linspace(-0.65,0.4,500),np.linspace(-0.7,0.6,500))   # get grid coordinates
    z=model.predict(np.c_[xx.flatten(),yy.flatten()])                       # predict which type each point in the grid belongs to
    zz=z.reshape(xx.shape)
    plt.contour(xx,yy,zz,[1])             # plot the decision boundary

np.set_printoptions(threshold=np.inf)
data = sio.loadmat('../data/ex6data3.mat')
x=data['X']        # x 211*2
y=data['y']        # y 211*1
xval=data['Xval']  # xval 200*2
yval=data['yval']  # yval 200*1

# choose a suitable C and gamma
C_values=[0.01,0.03,0.1,0.3,1,3,10,30,100]         # C bigger, easier to be overfitted.
gammas=[0.01,0.03,0.1,0.3,1,3,10,30,100]           # gamma bigger, easier to be overfitted.

best_score=0
best_parameters=(0,0)

for c in C_values:
    for g in gammas:
        svc1=svm.SVC(C=c,kernel='rbf',gamma=g)     # create a svm with linear kernel, C=1, C is the parameter of penalty
        svc1.fit(x,y.flatten())                    # train the svm, using train set data
        score = svc1.score(xval,yval.flatten())    # rate the model,using cross validation set data
        if score>best_score:
            best_score=score                       # note the better score and parameters
            best_parameters=(c,g)

fig=plt.figure(figsize=(8,6))
fig.canvas.set_window_title('Best parameters')

svc_best=svm.SVC(C=best_parameters[0],kernel='rbf',gamma=best_parameters[1])
svc_best.fit(x,y.flatten())
print('The best score=',best_score,'best parameters=',best_parameters) # print the accurancy of svc_best

plotBoundary(svc_best)                             # plot decision boundary
plotraw()
plt.show()
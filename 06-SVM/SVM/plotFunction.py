import numpy as np
import matplotlib.pyplot as plt

def plotraw(x,y):
    plt.scatter(x[:,0],x[:,1],c=y.flatten(),cmap='bwr')
    plt.xlabel('x1')
    plt.ylabel('x2')

def plotBoundary(model):
    xx,yy=np.meshgrid(np.linspace(-0.5,4.5,500),np.linspace(1.3,5,500))   # get grid coordinates
    z=model.predict(np.c_[xx.flatten(),yy.flatten()])                     # predict which type each point in the grid belongs to
    zz=z.reshape(xx.shape)
    plt.contour(xx,yy,zz,[1])       # plot the decision boundary
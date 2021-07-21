import numpy as np
import matplotlib.pyplot as plt
from gaussion import gaussion

def plotContour(mean,var,flag=0):
    x=np.linspace(0,30,100)
    y=np.linspace(0,30,100)
    xx,yy=np.meshgrid(x,y)                                       # the points of grid

    z=gaussion(np.c_[xx.flatten(),yy.flatten()],mean,var,flag)   # get probability
    zz=z.reshape(xx.shape)                                       # important
    levels=[10**h for h in range(-20,0,3)]                       # you can change 10**h to n**h for different space
    plt.contour(xx,yy,zz,levels)
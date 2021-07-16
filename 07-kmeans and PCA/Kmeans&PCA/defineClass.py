import numpy as np
from math import sqrt

def defClass(x,center):
    m=len(x)
    class_number=len(center)                     # the number of classes
    class_all=np.zeros(m)

    for i in range(0,m):
        dis=100000
        for j in range(class_number):

            temp=x[i,:]-center[j,:]                   # allocate each sample's class according to the distance between points and centers
            dis_j=sqrt(np.linalg.norm(temp))   # use the norm function to get distance

            if dis_j<dis:
                dis=dis_j
                class_all[i]=j                   # note the smaller distance class

    return class_all
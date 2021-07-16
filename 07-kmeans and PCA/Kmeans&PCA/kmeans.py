import numpy as np
from defineClass import defClass
from compCenters import compCenters


def kmeans(x,k,iters):
    initial_center_number=np.random.choice(len(x),k)      # k is the number of classes
    initial_center=np.zeros((k,x.shape[1]))
    for i in range(k):
        initial_center[i,:]=x[initial_center_number[i],:] # random choice 3 initial center
        initial_center=np.array(initial_center)

    center_note=np.zeros((k,x.shape[1],iters))            # note centers of some steps
    class_all=defClass(x,initial_center)                  # allocat each sample's class according to the initial center

    for i in range(0,iters):
        centers=compCenters(x,class_all)                  # iterate the mean center of each class
        class_all=defClass(x,centers)                     # iterate the class
        center_note[:,:,i]=centers                        # note the centers

    return centers,class_all,center_note
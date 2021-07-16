import numpy as np

def compCenters(x,class_all):
    class_list=np.unique(class_all)                  # get the list of classes

    centers=np.zeros((len(class_list),x.shape[1]))   # initialize a ndarray to store the mean centers

    for nn in class_list:
        temp=np.mean(x[class_all==nn,:],axis=0)      # calculate each class's mean center
        centers[int(nn),:]=temp

    return centers
import numpy as np

def meanNormalize(y,r):
    mean_y=np.sum(y,axis=1)/np.sum(r,axis=1)    # just use the score of movies that has been rated to calculate mean value
    mean_y=mean_y.reshape(-1,1)
    y=(y-mean_y)*r
    return y,mean_y
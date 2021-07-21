import numpy as np

def groupParams(x,theta):
    'group x and theta to a 1D data'
    return np.r_[x.flatten(),theta.flatten()]           # flatten is to make it 1D, so that can adapted the minimize function
    
def ungroupParams(data,n_movies,n_users,n_features):
    'divide data to x and theta'
    x=data[:n_movies*n_features]
    theta=data[n_movies*n_features:]
    
    x=x.reshape(n_movies,n_features)
    theta=theta.reshape(n_users,n_features)
    return x,theta
import numpy as np

def sig(z):
    return 1/(1+np.exp(-z))

def sig_grd(z):
    return sig(z)*(1-sig(z))
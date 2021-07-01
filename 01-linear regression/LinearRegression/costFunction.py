import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def costFunction(predictValue,y):
    temp=np.dot((predictValue-y),(predictValue-y).T)
    Jtheta=(temp)/(2*len(y))
    return Jtheta
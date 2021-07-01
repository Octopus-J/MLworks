import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def hypothesis(theta,x):
    hypothesis_resault =theta*x      # htheta=theta0+theta1*x1
    predictValue=np.sum(hypothesis_resault,axis=1)
    return predictValue
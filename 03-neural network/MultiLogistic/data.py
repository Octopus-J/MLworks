import numpy as np
import matplotlib.pyplot as plt
import scipy.io as asio


def getData(path):
    if path==' ':
        print("You didn't select the data path,try again.")
    else:
        data=2
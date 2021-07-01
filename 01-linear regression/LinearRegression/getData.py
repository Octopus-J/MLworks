from os import path
import pandas as pd

def getData():
    """
    get raw data
    """
    data=pd.read_csv(path,names=['population','profit'])
    return data

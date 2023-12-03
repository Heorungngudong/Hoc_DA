import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.linear_model import LinearRegression

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

def funtion_name():
    data = pd.read_csv("D:/Ngoc_Anh_ne/DA_T5/IOT/IOT-temp.csv")
    return data

#xuat file data
if __name__ == '__main__':
    data = pd.read_csv("D:/Ngoc_Anh_ne/DA_T5/IOT/IOT-temp.csv")
    data
    
    #xoa hai cot du lieu khong can thiet
    data = data.drop(['id','room_id/id'], axis = 1)
    
    
    
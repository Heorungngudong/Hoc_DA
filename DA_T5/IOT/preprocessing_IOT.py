
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def funtion_name():
    data = pd.read_csv("D:/Ngoc_Anh_ne/DA_T5/IOT/IOT-temp.csv")
    return data

#xuat file data
if __name__ == '__main__':
    data = pd.read_csv("D:/Ngoc_Anh_ne/DA_T5/IOT/IOT-temp.csv")
    data.head()
    
    #thong ke mo ta data
    data.describe()
    
    #kiem tra data co bi mat du lieu khong?
    data.isna()
    
    # chuan hoa du lieu bang ham boxcox
    normalized_data = stats.boxcox(data['temp'])
    # ve chart
    fig, ax=plt.subplots(figsize=(5, 3))
    sns.histplot(normalized_data, ax=ax, kde=True, legend=False)
    ax.set_title("Normalized Data")
    plt.show()
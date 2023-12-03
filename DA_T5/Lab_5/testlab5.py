import time, os, psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pyfpgrowth
import sys
import argparse


# doc file dat thanh list
def read_file_dat(file_name):
    #datContent = [i.strip().split() for i in open(f"D:/Ngoc_Anh_ne/DA_T5/Lab_4/{file_name}.dat").readlines()]
    datContent = [i.strip().split() for i in open(f"E:/IT/stock/{file_name}.dat").readlines()]
    return datContent

def apriori_find_frequent_itemsets(datContent, file_name, request_type): #ham dung thuat toan apriori de tim tap pho bien
    print('Thuat toan Apriori')
    te = TransactionEncoder()
    #danh sach minsupp
    listminsup = [0.9, 0.85, 0.8, 0.7, 0.6]
    listminconf = [0.9, 0.85, 0.8, 0.7, 0.6]
    list_timecost = []
    list_memorycost = []
    
    f = open(f'E:/IT/stock/{file_name}_apriori{request_type}.txt', 'w')
    for val in listminsup:
        for valconf in listmiconf:
            if request_type == 2:
                new_datContent = sort_dataset(datContent)
            elif request_type == 3:
                new_datContent = filter_dataset_item_minsup(datContent, val)
            else:
                new_datContent = datContent
            #chuyen doi ma tran du lieu thanh nhi phan
            te_arr = te.fit(new_datContent).transform(new_datContent)
            df = pd.DataFrame(te_arr, columns=te.columns_)
            # tim tap pho bien
            start_time = time.time() # bat dau tinh thoi gian chay
            frequent_itemsets = apriori(df, min_support=val, use_colnames=True, verbose=1, low_memory=True)
            end_time = time.time() # ket thuc thoi gian chay
        
            association_rule = association_rules (frequent_itemsets, min_support=val, min_confidence=minconf, metric="confidence",  use_colnames=True)

        # Tinh thoi gian chay va tieu thu bo nho
        timecost = end_time - start_time
        memorycost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        list_memorycost.append(memorycost)
        list_timecost.append(timecost)
        print('write to file')
        f.writelines(f"min_supp {val}")
        f.writelines(f"thoi gian thuc hien {timecost}s")
        f.writelines(f"bo nho tieu thu {memorycost} MB")
        f.writelines("Tap pho bien")
        f.writelines(" ")
        f.writelines(str(frequent_itemsets.head().values.tolist()))
        f.writelines(str(frequent_itemsets.tail().values.tolist()))
        f.writelines("Kich thuoc")
        f.writelines(str(frequent_itemsets.shape))
        print('Kich thuoc: ', frequent_itemsets.shape)
        #print(frequent_itemsets)
        
    f.close() 
    print(list_timecost)
    print(list_memorycost)

    # ve do thi
    plt.plot(list_timecost, list_memorycost, color='red', label='Thời gian và bộ nhớ tiêu thụ')
    plt.xlabel('Thời gian thực hiện (second)')
    plt.ylabel('Bộ nhớ tiêu tốn (MB)')
    plt.title('Biểu đồ thể hiện thời gian thực thi/bộ nhớ tiêu thụ')
    plt.legend()
    plt.show()

def fpgrowth_find_frequent_itemsets(datContent,file_name, request_type): #ham dung thuat toan fpgrowth de tim tap pho bien
    print('Thuat toan FP-Growth')
    #sys.setrecursionlimit(100)
    #danh sach minsupp
    listminsup = [0.9, 0.8, 0.7, 0.6, 0.55]
    list_timecost = []
    list_memorycost = []
    total_transaction = len(datContent)
    f = open(f'E:/IT/stock/{file_name}_fpgrowth.txt', 'w')
    for val in listminsup:
        if request_type == 2:
            new_datContent = sort_dataset(datContent)
        elif request_type == 3:
            new_datContent = filter_dataset_item_minsup(datContent, val)
        else:
            new_datContent = datContent
        threshold = val*total_transaction
        print(threshold)
        # tim tap pho bien
        start_time = time.time() # bat dau tinh thoi gian chay
        frequent_itemsets = pyfpgrowth.find_frequent_patterns(new_datContent, threshold)
        end_time = time.time() # ket thuc thoi gian chay

        # Tinh thoi gian chay va tieu thu bo nho
        timecost = end_time - start_time
        memorycost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        list_memorycost.append(memorycost)
        list_timecost.append(timecost)
        #print(frequent_itemsets[:5])
        print('write to file')
        f.writelines(f"min_supp {val}\n")
        f.writelines(f"thoi gian thuc hien {timecost}s\n")
        f.writelines(f"bo nho tieu thu {memorycost} MB\n")
        f.writelines("Tap pho bien\n")
        f.writelines(" ")
        f.writelines(str(list(frequent_itemsets)[:5]))
        f.writelines("Kich thuoc: ")
        print('Kich thuoc: ', len(frequent_itemsets))
        f.write(str(len(frequent_itemsets)))
     
    print(list_timecost)
    print(list_memorycost)
    f.close()
    # ve do thi
    plt.plot(list_timecost, list_memorycost, color='red', label='Thời gian và bộ nhớ tiêu thụ')
    plt.xlabel('Thời gian thực hiện (second)')
    plt.ylabel('Bộ nhớ tiêu tốn (MB)')
    plt.title('Biểu đồ thể hiện thời gian thực thi/bộ nhớ tiêu thụ')
    plt.legend()
    plt.show()
        
def sort_dataset(datContent):
    df = pd.DataFrame(datContent) #gan data bang bien df
    item_count = {} #tao dict dem item
    #vong lap dem item 
    for transaction in datContent:
        for item in transaction:
            if item in item_count:
                item_count[item] += 1
            else:
                item_count[item] = 1
    #sap xep cac item theo do pho bien
    sorted_list = sorted(item_count.keys(), key=lambda item: item_count[item])
    #tao data moi tu cac item duoc sap xep theo do pho bien
    sorted_df = df.apply(lambda row : sorted(row, key=lambda item : sorted_list.index(item)), axis=1)
    
    return sorted_df.to_list()


def filter_dataset_item_minsup(datContent, min_supp):
    total_transaction = len(datContent)
    #print('truoc khi loc: ', total_transaction)
    item_count = {} #tao list dem item
    item_supp = [] #tao list dem sup
    #vong lap dem item 
    for transaction in datContent:
        for item in transaction:
            if item in item_count:
                item_count[item] += 1
            else:
                item_count[item] = 1
    #print('So luong item Truoc khi tinh support', len(item_count))
    item_count.update((item, count/total_transaction) for item, count in item_count.items() )
    for key, count in item_count.items():
        if count >= min_supp:
            #print('Ket qua so sanh: ', count >= min_supp)
            item_supp.append(key)
    #print('So luong item Truoc khi tinh support', len(item_supp))
    new_datContent = []
    for transaction in datContent:
        for item in item_supp:
            if item in transaction:
                new_datContent.append(transaction)
                break
    #print('Truoc khi tinh support', len(new_datContent))
    return new_datContent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter request type')
    parser.add_argument('--request_type', type=int, required=True)
    args = parser.parse_args()
    
    list_file_name = ['chess','mushroom']
    for file_name in list_file_name:
        print(file_name)
        #print('So luong data goc: ', len(read_file_dat(file_name)))
        apriori_find_frequent_itemsets(read_file_dat(file_name), file_name, args.request_type)
        fpgrowth_find_frequent_itemsets(read_file_dat(file_name), file_name, args.request_type)
    


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time, os, psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import multiprocessing 
import pandas as pd

def readDATfile(file_name):
    datContent = [i.strip().split() for i in open(f"./{file_name}.dat").readlines()]
    return datContent
def convertDATtoDF(datContent):
    #chuyen doi ma tran du lieu thanh nhi phan
    te = TransactionEncoder()
    te_arr = te.fit(datContent).transform(datContent)
    df = pd.DataFrame(te_arr, columns=te.columns_)
    return df

def const_supp_val_conf(dat, request_type=None):
    print('Thuat toan Apriori')
    
    #danh sach minsupp
    listminsup = [0.9, 0.85, 0.8, 0.7, 0.6]
    listconf = [0.9, 0.85, 0.8, 0.75, 0.7]
    total_list_timecost = {}
    total_list_memorycost = {}
    for val1 in listminsup:
        if request_type == 3:
            df = convertDATtoDF(filter_dataset_item_minsup(dat, val1))
        else:
            df = convertDATtoDF(dat)
        list_timecost = []
        list_memcost = []
        for val2 in listconf:
            # tim tap pho bien
            start_time = time.time() # bat dau tinh thoi gian chay
            frequent_itemsets = apriori(df, min_support=val1, use_colnames=True, low_memory=True)

            end_time = time.time() # ket thuc thoi gian chay

            # Tinh thoi gian chay va tieu thu bo nho
            timecost = end_time - start_time
            memorycost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

            rules = association_rules(frequent_itemsets, metric = "confidence", support_only=True, min_threshold=val2)
            #rules = association_rules(frequent_itemsets, metric="confidence", support_only = True,  min_threshold=val2)
            list_timecost.append(timecost)
            list_memcost.append(memorycost)
            print(frequent_itemsets)
            print(rules)

            print('thoi gian: ',timecost)
            print('bo nho: ',memorycost)
        total_list_timecost[val1] = list_timecost
        total_list_memorycost[val1] = list_memcost

def const_conf_val_supp(dat, request_type=None):

    print('Thuat toan Apriori')

    #danh sach minsupp
    listminsup = [0.9, 0.85, 0.8, 0.7, 0.6]
    listconf = [0.9, 0.85, 0.8, 0.75, 0.7]
    total_list_timecost = {}
    total_list_memorycost = {}
    for val1 in listconf:
        list_timecost = []
        list_memcost = []
        for val2 in listminsup:
            if request_type == 3:
                df = convertDATtoDF(filter_dataset_item_minsup(dat, val2))
            else:
                df = convertDATtoDF(dat)
            # tim tap pho bien
            start_time = time.time() # bat dau tinh thoi gian chay
            frequent_itemsets = apriori(df, min_support=val2, use_colnames=True, low_memory=True)

            end_time = time.time() # ket thuc thoi gian chay

            # Tinh thoi gian chay va tieu thu bo nho
            timecost = end_time - start_time
            memorycost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


            rules_minsupchange = association_rules(frequent_itemsets, metric="confidence", support_only = True,  min_threshold=val1)
            
            list_timecost.append(timecost)
            list_memcost.append(memorycost)
            print( frequent_itemsets)
            print(rules_minsupchange)

            print('thoi gian: ',timecost)
            print('bo nho: ',memorycost)
        total_list_timecost[val1] = list_timecost
        total_list_memorycost[val1] = list_memcost

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--request_type", default=1, type=int)
    args = parser.parse_args()
    num_core = multiprocessing.cpu_count()
    if num_core >=10:
        num_workers = num_core-4
    else:
        num_workers = num_core
    # pool = multiprocessing.Pool(num_workers)
    list_file = ['chess', 'mushroom']
    for file_name in list_file:
        if args.request_type ==  1:
            const_supp_val_conf(convertDATtoDF(readDATfile(file_name)))
            const_conf_val_supp(convertDATtoDF(readDATfile(file_name)))
        elif args.request_type == 2:
            datContent = sort_dataset(readDATfile(file_name))
            const_supp_val_conf(convertDATtoDF(datContent))
            const_conf_val_supp(convertDATtoDF(datContent))
        elif args.request_type == 3:
            const_supp_val_conf(readDATfile(file_name), args.request_type)
            const_conf_val_supp(readDATfile(file_name), args.request_type)

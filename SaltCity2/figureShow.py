# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data():
    train_csv = 'dataset/train_20180125.csv'
    test_csv = 'dataset/test_20180125.csv'
    train_org = pd.read_csv(train_csv)
    test_org = pd.read_csv(test_csv)
    return train_org,test_org

def translate_date(x):
    xx = []
    for xi in x:
        xx.append((xi/100 - 2012)*12 + xi % 100)
    return xx

def draw_pic_month_totle(x,y):
    
    plt.plot(x,y)
    for i, xi in enumerate(x):
        #11月份数据标红
        if xi % 12 == 11:
            plt.scatter(xi,y[i],c='r')
    plt.grid()
    plt.xticks((0,12,24,36,48,60),('2012','2013','2014','2015','2016','2017'))
    plt.savefig('../pic/saledate_pic/month_totle_quantity.png',dpi = 300)
    plt.close('all')

def month_totle_quantity(train_org):
    saledate_set = set(train_org['sale_date'])
    sale_data_quantity = []
    for saledate in saledate_set:
        df_saledata = train_org[train_org['sale_date'] == saledate]
        quantity = df_saledata['sale_quantity'].sum()
        sale_data_quantity.append([saledate,quantity])
    sale_data_quantity.sort()
    x = []
    y = []
    for sale in sale_data_quantity:
        x.append(sale[0])
        y.append(sale[1])
    xx = translate_date(x)
    draw_pic_month_totle(xx,y)
    #print sale_data_quantity

def sort_month_quantity(train_org):
    '''
        对2017年平均月销量进行统计，查看前几个月的月销量对总销量的影响
    '''
    dict_class_quantity = defaultdict(lambda: 0)
    train_2017 = train_org[train_org['sale_year'] == 2017]
    class_ids = set(train_2017['class_id'])
    #class_quantity = []
    for class_id in class_ids:
        train_2017_class = train_2017[train_2017['class_id'] == class_id]
        cnt = train_2017_class.shape[0]
        quantity = train_2017_class['sale_quantity'].sum()
        #class_quantity.append([class_id,quantity/cnt])
        dict_class_quantity[class_id] = quantity/cnt
    #class_quantity.sort(key = lambda x:x[1],reverse = True)
    #print dict_class_quantity
    return dict_class_quantity
def separate_train_test(train_org,test_org,dict_class_quantity):
    '''
        将训练集分成两个部分，一部分是销量比较高的
    '''
    average_quantity = []
    for class_id in train_org['class_id']:
        average_quantity.append(dict_class_quantity[class_id])
    train_org['average_quantity_2017'] = average_quantity
    df_train1 = train_org[train_org['average_quantity_2017'] >= 400]
    df_train2 = train_org[train_org['average_quantity_2017'] < 400]
    df_train1.drop(['average_quantity_2017'],axis = 1,inplace=True)
    df_train2.drop(['average_quantity_2017'],axis = 1,inplace=True)
    df_train1.to_csv('dataset/train_20180130_1.csv',index = False)
    df_train2.to_csv('dataset/train_20180130_2.csv',index = False)

    average_quantity = []
    for class_id in test_org['class_id']:
        average_quantity.append(dict_class_quantity[class_id])
    test_org['average_quantity_2017'] = average_quantity
    df_test1 = test_org[test_org['average_quantity_2017'] >= 400]
    df_test2 = test_org[test_org['average_quantity_2017'] < 400]
    df_test1.drop(['average_quantity_2017'],axis = 1,inplace=True)
    df_test2.drop(['average_quantity_2017'],axis = 1,inplace=True)
    df_test1.to_csv('dataset/test_20180130_11.csv',index = False)
    df_test2.to_csv('dataset/test_20180130_22.csv',index = False)
    
if __name__ == '__main__':
    train_org,test_org = load_data()
    #month_totle_quantity(train_org)
    dict_class_quantity = sort_month_quantity(train_org)
    #print dict_class_quantity
    separate_train_test(train_org,test_org,dict_class_quantity)

    
    
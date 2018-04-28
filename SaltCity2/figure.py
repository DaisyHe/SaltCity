# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:48:11 2018

@author: Administrator
"""


from preprocess import getTrainData,sepTargetAndOther
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#作图    
def drawFig(x,y,title,xlabel,ylabel):
    plt.figure(figsize=(10,5))
    plt.plot(x,y,'ro') 
    
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title('Random Scatter')  
    
    plt.grid(True)  
    plt.savefig('imag.png')  
    plt.show()  

#年销售量统计
def year_sale(df_train):
    sale_years = [2012,2013,2014,2015,2016]
    class_ids = set(df_train['class_id'])
    x_sum = class_ids
    y_sum = []
    for year in sale_years:
        print 'year is %d' % year
        
        year_df = df_train[ df_train['sale_year'] == year ]
        print year_df.shape
        y = year_df.sum()
        print y.shape
        y_sum.append(y)
        
    drawFig(x_sum,y_sum,title = 'date_sale',xlabel='sale_date',ylabel='sale_quantity')    

if __name__ == '__main__':
    df_train = pd.read_csv('dataset/test_20180125.csv')

    #年销量对比
    #year_sale(df_train)#还不完善
   
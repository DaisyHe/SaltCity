# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 16:53:48 2018

@author: Administrator
"""


#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

from SaltCity2 import getTrainData,sepTargetAndOther,setMissinglevel_id,pred_score,monthIs11
from SaltCity2 import scaler,normalization,statistics,discrete,histgram
from SaltCity2 import modelTuningParams,plotLearningCurve,testModel,testXGBoost,testGBDT
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
from scipy.stats import boxcox


def getData(csvPath):
    
    #获取所有数据
    df_train_all = getTrainData(csvPath)
    
    #notBeUsed是筛选出来，没有进行学习的列
    notBeUsed = ['type_id','level_id','department_id','TR','gearbox_type','rated_passenger',
                 'if_charging','price_level','price','fuel_type_id','newenergy_type_id',
                 'emission_standards_id','if_MPV_id','if_luxurious_id','power','engine_torque',
                 #上面是做过one-hot的字段和非数值字段，下面为特征筛选
                 'front_track','rear_track','cylinder_number','displacement','driven_type_id',
                 'wheelbase']
    df_data = getTrainData(csvPath,notBeUsed)#获取部分列数据
    
    #如果是有序的，用shuffle函数打乱
    #train_org_shuffle = shuffle(df_train)
    col = df_data.columns.values
    #print col
    
    return df_data #or return df_train_all

def boxcox(df_train):
    sale_quantity = df_train['sale_quantity']
    sale_box,lambda_ = boxcox(sale_quantity)
    return sale_box

if __name__ == '__main__':
    ''' 读取数据 '''
    df_train_raw = getData('dataset/train_20180130_2.csv')
    
    #boxcox变换
    train_boxcox = boxcox(df_train_raw)
    df_train_raw['sale_quantity'] = train_boxcox
    
    df_train = shuffle(df_train_raw)
    #columns = df_train.columns.values
    #for col in columns:
        #print col
    
    df_test_all = getData('dataset/test_20180130_22.csv')
    df_test = df_test_all.drop('sale_quantity',axis = 1)    #测试集中，需去掉销售量这一列
    #将数据分为训练数据和目标数据
    
    df_train_data,df_train_target = sepTargetAndOther(df_train)
    X = df_train_data
    y = df_train_target
    print len(X),len(y),len(df_test)
    
    print type(df_test)
    class_id_X = df_test['class_id']
    set_train = set()
    print len(set_train)
    '''数据预处理与可视化'''
    #数据的预处理在preprocess.py(贺)和preprocessRen.py(任)中进行处理
    #缺失数据处理
    #df_train,rfr = setMissinglevel_id(df_train_all)
    
    #观察每年11月份的销售量
    #path = monthIs11(df_train)
    #monthIs11 = getTrainData(path)
       
    '''特征选择'''
    #featureSelection.py,该文件选取单个特征，组合特征的选取，温在做
    
    '''模型及调参（利用学习曲线）'''
    #测试，显示哪一个模型效果好
    testModel(X,y,df_test,save=True,fileName='predict_RFR_all2_') #目前根据模型结果，随机森林回归的效果最好 ，需要进行调参
    #testXGBoost(X,y,df_test,save=False,fileName='predict_XGBoost_all_')
    #testGBDT(X,y,df_test,save=True,fileName='predict_GBDT_all_')
    #模型调参
    #modelTuningParams(X,y)
    #注：后期可以用 模型融合 提高结果精度
    '''模型融合'''




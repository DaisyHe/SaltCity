# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 16:27:36 2018

@author: Administrator
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import numpy as np
from pandas import DataFrame
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.stats import boxcox

#读取数据
def getTrainData(csvPath,notBeUsed = []):
    #df_train = DataFrame.from_csv(csvPath,header = 0)
    df_train = DataFrame.from_csv(csvPath,header = 0)
    #print df_train.columns.values
    #df_train_origin = df_train #保存原始数据
    #df_train.head(10) #查看前10条数据
    
    #print '字段名称与类型：','\n',df_train.dtypes
    #print '数据集大小：','\n',df_train.shape
    #print '列统计：','\n',df_train.count()
    
    #df_train['year'] = pd.DatetimeIndex(df_train.sale_date).year
    #df_train['month'] = pd.DatetimeIndex(df_train.sale_date).month
    df_train = df_train.drop(notBeUsed,axis = 1)
    return df_train
    
#将目标字段和其余字段分离开
def sepTargetAndOther(df_train):
    df_train_target = df_train['sale_quantity'].values #目标数据
    df_train_data = df_train.drop(['sale_quantity'],axis = 1).values#除去目标数据之外的数据
    return df_train_data,df_train_target

#设置缺失值（level_id）——该函数不正确
def setMissinglevel_id(df):
    print df.ix[:10,'level_id']
    level_id_df = df[['level_id','type_id','department_id','if_luxurious_id','car_length','car_width','car_height','wheelbase','front_track','rear_track']]
    
    print level_id_df.level_id.shape  #(20157L,)
    #for levelid in level_id_df.level_id:
        #print levelid
    known_level_id = level_id_df[[levelid for levelid in level_id_df.level_id if levelid is not '-']].as_matrix()
    unknown_level_id = level_id_df[[levelid for levelid in level_id_df.level_id if levelid is '-']].as_matrix()
    
    y = known_level_id[:,0] #目标年龄
    X = known_level_id[:,1:] #特征属性值
    
    #fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state = 0,n_estimators = 2000, n_jobs = -1)
    rfr.fit(X,y)
    
    #用得到的模型进行未知level_id的结果预测
    predicted_level_id = rfr.predict(unknown_level_id[:,1::])
    
    df.loc[(df.level_id.isnull()),'level_id'] = predicted_level_id
    df.ix[:,'level_id'] = df.ix[:,'level_id'].astype('int64')
        
    return df,rfr

        
#筛选出月份是11的数据并按年月顺序进行保存
def monthIs11(df_data):
    #获得数据字段
    columns = df_data.columns.values
    #print columns.shape
    
    class_id_set = set(df_data['class_id'])
    print len(class_id_set)
    
    ym = [201211,201311,201411,201511,201611]
    data_11 = DataFrame(np.random.randn(0,116),columns=columns)
    for y in ym:
        d1 = df_data[ df_data['sale_date'] == y ]
        data_11 = pd.merge_ordered(d1,data_11)  #为什么merge后，行数减少了
    path = 'dataset/monthis11.csv'
    data_11.to_csv(path)
    return path
    
#


''' 数值型数据处理的方法 '''
#幅度调整到一定的范围
def scaler(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_min_max = min_max_scaler.fit_transform(data)
    return data_min_max
    
#归一化
def normalization(data):
    #data_scale = preprocessing.scale(data)
    #mean = data_scale.mean(axis=0)
    #std = data_scale.std(axis=0)
    #print 'mean:%0.3f,std:%0.3f'%(mean,std)
    
    scaler = preprocessing.StandardScaler().fit(data)
    #print scaler
    mean_ = scaler.mean_
    scale_ = scaler.scale_
    data_scaler = scaler.transform(data)
    print 'standardScaler::mean:%0.3f,std:%0.3f'%(mean_,scale_)
    print data_scaler
    
    return data_scaler
    
#统计
def statistics(data):
    #col_mean = data.mean(0)#求均值
    series = pd.Series(data)
    series.describe(percentiles=[.05,.25,.75,.95])
   
#离散化
def discrete(df_data,used):#可以是区间或段数，如[-1,2,4,5]或4段
    data = df_train[used]
    #rangeOrNumber = np.arange(0,data.max()+1,1)
    range_data = pd.cut(data,4)
    print range_data#cut之后应该是class 'pandas.core.categorical.Categorical'才对，但是我的仍然是Series，不解
    #disc_data = pd.value_counts(range_data).reindex(range_data.levels)#Categorical才有levels属性，所以现在只能用下面的笨方法进行填充
    length_level = []
    
    #length:4072.75 4470.5 4868.25 
    #width:1618 1726 1834 
    #engine_torque:152.5 235 317.5 
    #total_quality:1641.25 1932.5 2223.75
    #equipment_quality:1198.75 1452.5 1706.25 
    #wheelbase:2547 2734 2921 
    #front_track:1373.75 1467.5 1561.25
    #rear_track:1385.25 1480.5 1575.75
    for i in data:
        if i<=1385.25:
            level = 0
        elif i<=1480.5:
            level = 1
        elif i<=1575.75:
            level = 2
        else:
            level = 3
        length_level.append(level)  
    df_train['rear_track'] = length_level
    df_train.to_csv('dataset/train_discrete7.csv',index=False)
    return length_level,df_train

#柱状比例分布
def histgram(data):
    series = pd.Series(data)
    series.value_counts()#pd.value_counts(data)
    
#近两年趋势
def trend(df_train):
    print len(df_train)
    #得到201511~201610,201611~201710的数据
    df_train_date1 = df_train[(df_train.sale_date>=201511)&(df_train.sale_date<=201610)]
    df_train_date2 = df_train[(df_train.sale_date>=201610)&(df_train.sale_date<=201710)]
    
    #得到所有的class——id
    class_ids = set(df_train['class_id'])
    #得到每个class_id的趋势
    trends = defaultdict(lambda:dict(trend=[]))
    for class_id in class_ids:
        sum1 = df_train_date1['class_id'].isin([class_id]).sum()
        sum2 = df_train_date2['class_id'].isin([class_id]).sum()
        if sum1 < sum2:
            trend = 1
        else:
            trend = 0
        trends[class_id]['trend'].append(trend)
    #对原始dataframe进行填充
    class_id_df = df_train['class_id']
    trends_list = []
    for class_id in class_id_df:
        df_trend = trends[class_id]['trend'][0]
        trends_list.append(df_trend)
    #将两年走势保存到csv文件
    df_train['trend'] = trends_list
    df_train.to_csv('dataset/test_trend.csv',index=False)

#做聚类
def cluster(df_train,used):
    #features = df_train.columns.values
    brand_id = df_train[used]
    
    feature = brand_id#.reshape(-1,1)单个特征  #.reshape(1,-1)单个样本
    #调用kmeans类
    clf = KMeans(n_clusters=4)
    s = clf.fit(feature)
    print s
    
    #4个中心
    print clf.cluster_centers_
    
    #每个样本所属的簇
    print clf.labels_
    df_train['cluster_brandId'] = clf.labels_
    df_train.to_csv('dataset/cluster_brandId.csv',index=False)
    
    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print clf.inertia_
    #进行预测
    predict = clf.predict(feature)
    print predict
    #保存模型
    joblib.dump(clf,'dataset/cluster.pkl')
    
    '''
    #载入保存模型
    clf = joblib.load('dataset/cluster.pkl')
    
    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    #但是目前聚类的结果是：簇数越多越好
    for i in range(5,36,1):
        clf = KMeans(n_clusters=i)
        s = clf.fit(feature)
        print i,clf.inertia_
    '''
    
#按照class_id进行划分，不过目前来看，新的一年中就有所有class_id
def class_id_split(df_train):
    #得到两个时间段的数据
    #df_train_date1 = df_train[(df_train.sale_date>=201201)&(df_train.sale_date<=201610)]
    df_train_date2 = df_train[(df_train.sale_date>=201610)&(df_train.sale_date<=201710)]
    class_ids = df_train['class_id']
    #class_id_set1 = set(df_train_date1)
    class_id_set2 = set(df_train_date2['class_id'])
    
    have_class_id = []
    for class_id in class_ids:
        if class_id in class_id_set2:
            have_id = 1
        else:
            have_id = 0
        have_class_id.append(have_id)
    df_train['new_have_class_id'] = have_class_id
    df_train.to_csv('dataset/circle_year_new.csv')    

if __name__ == '__main__' :
    '''
    df_train = pd.read_csv('dataset/train_discrete6.csv')
    
    #离散化，labels = [0,1,2,3],但是目前离散化的工具不能自动查看标签，区间是我用cut划分之后，然后按照cut划分的标准手动设置的
    #level_discrete,df_train_discrete = discrete(df_train,used='rear_track')
    '''
    
    df_train = pd.read_csv('dataset/test_20180125.csv')

    #年销量对比
    #year_sale(df_train)#还不完善
    
    #每个class_id在201511~201610和201611~201711的销量走势，并填充新字段(trend)
    trend(df_train)
    
    #按照brand_id进行聚类
    #cluster(df_train,used = ['brand_id','sale_quantity'])
    
    #统计2012~201610和201611~201710两个时间段的class_id
    #第二阶段有全部class_id，所以该函数无意义
    #class_id_split(df_train)
    #df_train_date2 = df_train[(df_train['sale_date']==201710)&(df_train['class_id']==379265)]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
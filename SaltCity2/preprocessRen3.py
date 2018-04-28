# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 08:54:36 2018

@author: HUI
"""
import numpy as np
import pandas as pd
import os

def getClassID(trainData):
    class_ids = set(trainData['class_id'])
    #print len(class_id)
    return class_ids

def getBrandID(trainData):
    brand_ids = set(trainData['brand_id'])
    print len(brand_ids)
    
    
def getSerParams(seri):
    _ids = set(seri)
    print _ids
    print '--->',len(_ids)
def getColumnsParams(trainData):
    columns = set(trainData.columns.values)
    for col in columns:
        _ids = set(trainData[col])
        print _ids
        print col,'--->',len(_ids)

def fromClassIDGetColumns(trainData):
    columns = set(trainData.columns.values)
    class_ids = getClassID(trainData)
    for class_id in class_ids:
        temp_columns = set()
        df_class_id = trainData[trainData['class_id'].isin([class_id])]
        #print df_class_id
        for col in df_class_id.columns.values:
            val = set(df_class_id[col])
            if len(val) == 1:
                temp_columns.add(col)
        columns = columns & temp_columns
        
    return columns

def load_data():
    #train_csv = 'data/yancheng_train_20171226.csv'
    train_csv = 'data/train1.csv'
    test_csv = 'data/yancheng_testA_20171225.csv'
    #train_org = pd.DataFrame.from_csv(train_csv)
    train_org = pd.read_csv(train_csv)
    features = train_org.columns.tolist()
    test_org = pd.read_csv(test_csv)
    #print train_columns
    numerics = ['int16','int32','int64','float16','float32','float64']
    features_numeric = train_org.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    
    return train_org,test_org,features,features_numeric,features_non_numeric
    
def process_non_numeric_data(train_org,features_non_numeric):
    #首先处理level_id，对应class_id=178529全 部为'-'
    train_org['level_id'] = train_org['level_id'].replace('-','6')
    train_org['level_id'] = train_org['level_id'].astype('int32')
    #处理TR字段,主要处理的是8;7,该字段总共有21条记录，全部处理为8
    train_org['TR'] = train_org['TR'].replace('8;7','3')#因为没有3
    train_org['TR'] = train_org['TR'].replace('5;4','2')#因为没有2
    train_org['TR'] = train_org['TR'].astype('int32')
    #处理字段gearbox_type，主要处理AT;DCT，处理为AT，然后对该字段进行one-hot编码,isinstance判断变量类型
    #train_org['gearbox_type'] = train_org['gearbox_type'].replace('AT;DCT','AT')
    #train_org['gearbox_type'] = train_org['gearbox_type'].replace('MT;AT','AT')
    train_org['gearbox_type_at']    = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'AT' in x else 0)
    train_org['gearbox_type_mt']    = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'MT' in x else 0)
    train_org['gearbox_type_dct']   = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'DCT' in x else 0)
    train_org['gearbox_type_cvt']   = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'CVT' in x else 0)
    train_org['gearbox_type_amt']   = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'AMT' in x else 0)
    train_org['gearbox_type_at_dct']= train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'AT;DCT' in x else 0)
    train_org['gearbox_type_mt_at'] = train_org.gearbox_type.apply(lambda x: 0 if isinstance(x, float) else 1 if 'MT;AT' in x else 0)
    #print train_org['gearbox_type_at']
    #处理字段 if_charging，做one-hot编码
    train_org['if_charging_l'] = train_org.if_charging.apply(lambda x: 0 if isinstance(x, float) else 1 if 'L' in x else 0)
    train_org['if_charging_t'] = train_org.if_charging.apply(lambda x: 0 if isinstance(x, float) else 1 if 'T' in x else 0)
    # 处理price_level字段，做one-hot编码
    train_org['price_level_5wl']    = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '5WL' in x else 0)
    train_org['price_level_5_8w']   = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '5-8W' in x else 0)
    train_org['price_level_8_10w']  = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '8-10W' in x else 0)
    train_org['price_level_10_15w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '10-15W' in x else 0)
    train_org['price_level_15_20w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '15-20W' in x else 0)
    train_org['price_level_20_25w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '20-25W' in x else 0)
    train_org['price_level_25_35w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '25-35W' in x else 0)
    train_org['price_level_35_50w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '35-50W' in x else 0)
    train_org['price_level_50_75w'] = train_org.price_level.apply(lambda x: 0 if isinstance(x, float) else 1 if '50-75W' in x else 0)
    #处理fuel_type_id，程序测试出来有8种，1L,2L,3L,4L,'1','2','3','-',所以先进行填充，然后类型转换
    train_org['fuel_type_id'] = train_org['fuel_type_id'].replace('-','1')
    train_org['fuel_type_id'] = train_org['fuel_type_id'].astype('int32')
    # 处理power，这里只处理脏数据，
    # 处理为‘81/70’为81，这里只有3列
    # 处理'-'为240纯属看价格来的，应该是用回归做出来的，而不应该是自己yy一个值
    train_org['power'] = train_org['power'].replace('81/70','81')
    train_org['power'] = train_org['power'].replace('-','240')
    train_org['power'] = train_org['power'].astype('float64')
    #处理engine_torque字段，该字段是发动机扭矩
    # 其中527765对应的全是‘-’，这个数据和扭矩缸数有关系，应该不会太大，后期需要好好处理一下，目前暂时处理为70，自己yy的一个数据
    # 处理155/140 为155，与pow对应
    train_org['engine_torque'] = train_org['engine_torque'].replace('-','70')
    train_org['engine_torque'] = train_org['engine_torque'].replace('155/140','155')
    train_org['engine_torque'] = train_org['engine_torque'].astype('float64')
    # 处理字段rated_passenger
    # 关于‘4月5日’的处理，根据class_id 883691的推断，应该是5，需要注意的是，4月5号，该编码格式无法识别
    # 后期优化的时候可以采用回归的方法进行预测，而不是仅仅通过观察，如通过发动机功率和车的长宽高进行预测
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('4\xd4\xc25\xc8\xd5','5')
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('5\xd4\xc27\xc8\xd5','7')
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('5\xd4\xc28\xc8\xd5','7')
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('6\xd4\xc27\xc8\xd5','7')
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('6\xd4\xc28\xc8\xd5','5')
    train_org['rated_passenger'] = train_org['rated_passenger'].replace('7\xd4\xc28\xc8\xd5','7')
    #getSerParams(train_org['rated_passenger'])
    train_org['rated_passenger'] = train_org['rated_passenger'].astype('int64')

    #print type(train_org['level_id'][10])

def process_columns_onehot(name,series_columns):
    #train_org['brand_id'] = train_org['brand_id'].astype('int32')
    brand_ids = set(series_columns)
    #print brand_ids
    for brand_id in brand_ids:
        columns_name = '{0}_{1}'.format(name,brand_id)
        #print brand_name
        train_org[columns_name] = series_columns.apply(lambda x: 1 if brand_id == x else 0)

def process_numeric_data(train_org,features_numeric):
    #处理日期sale_date字段，将日期处理为年份和月份
    train_org['sale_year'] = train_org.sale_date.apply(lambda x: x/100)
    train_org['sale_month'] = train_org.sale_date.apply(lambda x: x%100)
    # 处理brand_id品牌编号字段，不同编号之间应该没有大小关系，所以这里考虑one-hot编码
    # 命名格式为brand_id_XX,XX是对应的ID
    process_columns_onehot('brand_id',train_org['brand_id'])
    # 处理type_id车型类别字段
    process_columns_onehot('type_id',train_org['type_id'])
    # 处理level_id车型级别ID字段
    process_columns_onehot('level_id',train_org['level_id'])
    # 处理department_id，车型系别ID字段
    process_columns_onehot('department_id',train_org['department_id'])
    # 处理driven_type_id驱动形式ID字段
    process_columns_onehot('driven_type_id',train_org['driven_type_id'])
    # 处理newenergy_type_id，新能源类型ID字段
    process_columns_onehot('newenergy_type_id',train_org['newenergy_type_id'])
    # 处理emission_standards_id排放标准ID字段
    process_columns_onehot('emission_standards_id',train_org['emission_standards_id'])

def process_repeat_data(train_org):
    #获取列名称
    cols = train_org.columns
    print cols
    #初始化一个dataframe,会不会是这里的列编号问题？感觉不是===>事实证明，就是这里编号问题
    train = pd.DataFrame(columns = cols[1:])
    # 首先获取总共有多少个class_id
    class_ids = set(train_org['class_id'])
    # 获取 所有的销售日期
    sale_dates = set(train_org['sale_date'])

    #取销售编号为class_id的所有行组成的dataframe
    for class_id in class_ids:
        train_class_id = train_org[train_org['class_id'] == class_id]
        # 取销售日期为sale_date的所有列
        for sale_date in sale_dates:
            train_class_date = train_class_id[train_class_id['sale_date'] == sale_date]
            sale_sum = train_class_date['sale_quantity'].sum()
            #print sale_sum
            if  sale_sum != 0:
                #train_class_date.shape
                sale_month_max = train_class_date['sale_quantity'].max()
                train_row = train_class_date[train_class_date['sale_quantity'] == sale_month_max]
                train_row['sale_quantity'] = sale_sum
                #train.add(train_row)
                train = pd.merge_ordered(train,train_row)
                #print train.shape
    return train

def get_common_different_columns(train_class_id):
    #print train_class_id.shape
    common_columns = []         # 用于存储列中只有一个值的列
    different_columns = []      # 用于存储列中有多个值的列
    for col in train_class_id.columns.values:
        val = set(train_class_id[col])
        if len(val) == 1:
            common_columns.append(col)
        else:
            different_columns.append(col)
    return common_columns,different_columns
def get_common_columns_value(train_series):
    return set(train_series).pop()
def get_different_columns_value(df_train,train_column):
    # 首先要知道有那些字段
    vals = set(df_train[train_column])
    sum_val = -1
    ans = []
    for val in vals:
        # 提取val所在的行，组成一个新的dataframe
        df_val = df_train[df_train[train_column] == val]
        # 求对应val的销量和
        if sum_val < df_val['sale_quantity'].sum():
            sum_val = df_val['sale_quantity'].sum()
            ans = val
    return ans

def process_test_data(train_org, test_org):
    #print train_org.head()
    common_columns = []         # 用于存储列中只有一个值的列
    different_columns = []      # 用于存储列中有多个值的列
    # 首先获取训练集中所有列的名称
    train_columns = train_org.columns
    #print train_columns
    #test = pd.DataFrame(columns = train_columns)
    test_rows = []
    for test_class_id in test_org['class_id']:
        # 将编号为test_class_id的列提取出来，组成一个临时的dataframe
        df_temp = train_org[train_org['class_id'] == test_class_id]
        # 获取值相同的列和值不同的列
        common_columns,different_columns = get_common_different_columns(df_temp)
        #用于存储测试集每一行元素的值
        test_row_values = []
        # 对列进行遍历
        for train_column in train_columns:
            if train_column in common_columns:# 将值相同的列添加到测试集中
                test_row_values.append(get_common_columns_value(df_temp[train_column]))
            else: # 将值不同的列添加到测试机中
                test_row_values.append(get_different_columns_value(df_temp,train_column))
        # 将所有行的值组成一个二维list
        test_rows.append(test_row_values)
    # 将二维list转换成dataframe
    df_test = pd.DataFrame(test_rows, index = range(0,140), columns = train_columns)
    df_test.to_csv('data/test.csv')
    print df_test.shape

       
if __name__ == '__main__':
    # 加载数据
    train_org,test_org,features,features_numeric,features_non_numeric = load_data()
    '''
        处理训练数据
    '''
    # 处理重复数据
    #train_org = process_repeat_data(train_org)
    # 处理非数值型数据
    #process_non_numeric_data(train_org,features_non_numeric)
    # 处理数值型数据
    #process_numeric_data(train_org,features_numeric)
    #保存数据,在这里想办法删除第一列，不删除将会导致第一列重复
    #train_org.to_csv('data/train.csv')
    
    '''
        处理测试数据
    '''
    process_test_data(train_org,test_org)
    
    '''
        后面的代码暂时没用到
    '''
    #print features_non_numeric
    #print features_numeric
    
    #getColumnsParams(train_org)
    #getBrandID(trainData)
    #ans = fromClassIDGetColumns(trainData)
    #print ans
    #df_714860 = trainData[trainData['class_id'].isin([714860])]
    #for col in df_714860.columns.values:
    #    val = set(df_714860[col])
    #    if len(val) == 1:
    #        print col,'= ',val
    #    print'__________'
    #print type(trainData)
    #print trainData.head()
    #print trainData.shape
    #print trainData[trainData.isin.(['class_id'])]
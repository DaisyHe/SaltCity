# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:48:21 2018

@author: hejia
"""

import numpy as np
import pandas as pd

if __name__ == '__main__':
    predict1 = pd.read_csv('dataset/predict_GBDT_all1_.csv')
    predict2 = pd.read_csv('dataset/predict_RFR_all2_.csv')
    result = pd.read_csv('dataset/yancheng_testA_20171225.csv')
    p1_classid = list(predict1['class_id'])
    p2_classid = list(predict2['class_id'])
    class_ids = result['class_id']
    pre = []
    i=0
    j=0
    for class_id in class_ids:
        if class_id in p1_classid:
            p1 = predict1[ predict1['class_id'] == class_id ]
            p11 = p1.at[i,'predict_quantity']
            i += 1
            pre.append(p11)
        elif class_id in p2_classid:
            p2 = predict2[ predict2['class_id'] == class_id ]
            p22 = p2.at[j,'predict_quantity']
            j += 1
            pre.append(p22)    
    result['predict_quantity'] = pre
    result.to_csv('dataset/predict_20180130_GBDT_RFR.csv',index=False)
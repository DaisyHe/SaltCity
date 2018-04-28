# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:06:06 2018

@author: Administrator
"""

import math
import numpy as np

#得分函数
def pred_score(predict,actualValue):
    score = np.sqrt(np.mean((predict-actualValue)**2))
    return score

#和上面的函数一样，只是实现方法不一样
def pred_score1(predict,actualValue):
    n = len(predict)
    sum = 0

    #for p,a in predict,actualValue:
    for i in range(n):
        dp = (predict[i]-actualValue[i])**2
        sum += dp
    score = math.sqrt(sum/float(n))
    return score

if __name__ == '__main__':
    a = [1,2,3,4,5,6,7]
    b = [3,4,5,3,7,8,9]
    a = np.asarray(a)
    b = np.asarray(b)
    print pred_score1(a,b)
    print pred_score(a,b)
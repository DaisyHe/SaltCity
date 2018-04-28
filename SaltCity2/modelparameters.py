# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 22:13:25 2018

@author: Administrator
"""

from SaltCity2 import getTrainData,sepTargetAndOther,pred_score
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import xgboost as xgb
from sklearn.model_selection import KFold,train_test_split,GridSearchCV
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor

def testGBDT(X,y,df_test,save,fileName):
    #定义一个字典类型，用于存取几次预测的数据
    score_pred = defaultdict(lambda:dict(preds=[]))
    
    print 'GBDT：'
    result = getTrainData('dataset/test_20180130_1.csv')
    print type(result)
    kf = KFold(n_splits=2,shuffle=True,random_state=1234)
    
    for train_index,test_index in kf.split(X):
        gbdt_model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=500,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False).fit(X[train_index],y[train_index])
        pred = gbdt_model.predict(X[test_index])
        score = pred_score(pred,y[test_index])
        print "predict score: {0:.3f}".format(score)
        #print type(X[test_index]) #<type 'numpy.ndarray'>
        #对测试集进行预测
        #print type(df_test)#<class 'pandas.core.frame.DataFrame'>
        df_test = np.asarray(df_test) #这里将DataFrame才能进行预测，不知道是什么问题，随机森林的时候不需要转换类型，dataFrame直接可以用于测试
        predict = gbdt_model.predict(df_test)#.astype('int64')
        #将训练集预测份数和对应模型的预测结果保存到字典
        score_pred[score]['preds'] = predict
        #保存预测数据
    if save == True:
        min_score = sorted(score_pred.items(), key = lambda e:e[0])#按从小到大的顺序排列
        print min_score[0][0]#取得分最小的一个，第一个0为取最小key所对应的一行，第二个0为取score（最小）
        print score_pred[min_score[0][0]]['preds']
        result['predict_quantity'] = score_pred[min_score[0][0]]['preds']
        result.to_csv('dataset/%s1.csv' % fileName)
    '''
    #GBDT调参
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
    param_dict = {'max_depth':[2,4,6],'n_estimators':[50,100,200]}
    model = GridSearchCV(GradientBoostingRegressor(),param_dict)#寻找最好的参数设置
        model.fit(X_train,y_train)

        #最优模型
        print '最优模型：'
        print (model.best_estimator_)
        print ('得分为：')
        pred = model.predict(X_test)
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            error = pred_score(pred,y_test)
        print "predict error: {0:.3f}".format(error)
    '''

def testXGBoost(X,y,df_test,save,fileName):
    
    #定义一个字典类型，用于存取几次预测的数据
    score_pred = defaultdict(lambda:dict(preds=[]))
    
    print 'xgboost：'
    result = getTrainData('dataset/yancheng_testA_20171225.csv')
    test_result = ('dataset/')
    kf = KFold(n_splits=2,shuffle=True,random_state=1234)
    for train_index,test_index in kf.split(X):
        xgb_model = xgb.XGBRegressor(n_estimators=200,max_depth=4).fit(X[train_index],y[train_index])
        pred = xgb_model.predict(X[test_index])
        score = pred_score(pred,y[test_index])
        print "predict score: {0:.3f}".format(score)
        #print type(X[test_index]) #<type 'numpy.ndarray'>
        #对测试集进行预测
        #print type(df_test)#<class 'pandas.core.frame.DataFrame'>
        df_test = np.asarray(df_test) #这里将DataFrame才能进行预测，不知道是什么问题，随机森林的时候不需要转换类型，dataFrame直接可以用于测试
        predict = xgb_model.predict(df_test)#.astype('int64')
        #将训练集预测份数和对应模型的预测结果保存到字典
        score_pred[score]['preds'] = predict
        #保存预测数据
    if save == True:
        min_score = sorted(score_pred.items(), key = lambda e:e[0])#按从小到大的顺序排列
        print 111
        print min_score[0][0]#取得分最小的一个，第一个0为取最小key所对应的一行，第二个0为取score（最小）
        print 222
        print score_pred[min_score[0][0]]['preds']
        result['predict_quantity'] = score_pred[min_score[0][0]]['preds']
        result.to_csv('dataset/%s.csv' % fileName)
    '''
    #xgboost调参
    #模型
    xgb_model = xgb.XGBRegressor()
    #参数字典
    param_dict = {'max_depth':[2,4,6],'n_estimators':[50,100,200]}
    
    rgs = GridSearchCV(xgb_model,param_dict)
    rgs.fit(X,y)
    pred = xgb_model.predict(X[test_index])
    ground_truth = y[test_index]
    score = pred_score(pred,ground_truth)
    print score
    print rgs.best_params_
    '''

def testModel(X,y,df_test,save,fileName):
    
    #定义一个字典类型，用于存取几次预测的数据
    score_pred = defaultdict(lambda:dict(preds=[]))
    
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=3,test_size=0.2,random_state=0)
    result = getTrainData('dataset/test_20180130_2.csv')
    #print result['predict_quantity'].dtypes  #结果是float64型的，很奇怪
    print '随机森林回归/RandomForestRegressor(n_estimators = 100)'
    
    for train_index,test_index in cv:
        svc = RandomForestRegressor(n_estimators = 100).fit(X[train_index],y[train_index])
        pred = svc.predict(X[test_index])#预测训练数据
        score = pred_score(pred,y[test_index])#计算错误率
        print "predict score: {0:.3f}".format(score)
        #print type(X[test_index])#<type 'numpy.ndarray'>
        #print type(df_test)#<class 'pandas.core.frame.DataFrame'>
        #print X[test_index].shape,df_test.shape
        #对测试集进行预测
        predict = svc.predict(df_test)#.astype('int64')
        #将训练集预测份数和对应模型的预测结果保存到字典
        score_pred[score]['preds'] = predict
        #保存预测数据
    if save == True:
        min_score = sorted(score_pred.items(), key = lambda e:e[0])#按从小到大的顺序排列
        print min_score[0][0]#取得分最小的一个，第一个0为取最小key所对应的一行，第二个0为取score（最小）
        print score_pred[min_score[0][0]]['preds']
        result['predict_quantity'] = score_pred[min_score[0][0]]['preds']
        result.to_csv('dataset/%s.csv' % fileName)
    
    '''
    print '岭回归'
    for train,test in cv:
        svc = linear_model.Ridge().fit(df_train_data[train],df_train_target[train])
        pred=svc.predict(df_train_data[test])
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            score = pred_score(pred,df_train_target[test])
        print "predict score: {0:.3f}".format(score)
    
    print '支持向量回归/SVR(kernel='rbf',C=10,gamma=0.01)'
    for train,test in cv:
        svc = svm.SVR(kernel='rbf',C=10,gamma=0.01).fit(df_train_data[train],df_train_target[train])
        pred=svc.predict(df_train_data[test])
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            score = pred_score(pred,df_train_target[test])
        print "predict score: {0:.3f}".format(score)
    '''    
    '''print '随机森林回归/RandomForestRegressor(n_estimators = 100)'
    for train,test in cv:
        svc = RandomForestRegressor(n_estimators = 100).fit(df_train_data[train],df_train_target[train])
        pred=svc.predict(df_train_data[test])
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            score = pred_score(pred,df_train_target[test])
        print "predict score: {0:.3f}".format(score)'''
        
    '''
    #GBDT
    print "GBDT/loss='ls', learning_rate=0.1, n_estimators=100"
    for train, test in cv: 
        gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1, min_samples_split=2,
                                     min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, 
                                     alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
        gbdt.fit(df_train_data[train],df_train_target[train])
        pred=gbdt.predict(df_train_data[test])
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            score = pred_score(pred,df_train_target[test])
        print "predict score: {0:.3f}".format(score)  
    '''
    
    '''
    #判断是否过拟合
    estimator = RandomForestRegressor(n_estimators = 100)
    title = "Learning Curves (Random Forest, n_estimators = 100)"
    plot = plotLearningCurve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)
    plot.show()
        
    #缓解过拟合_尝试用不同的参数进行训练
    print "随机森林回归/Random Forest(n_estimators=200, max_features=0.6, max_depth=15)"
    for train, test in cv: 
        svc = RandomForestRegressor(n_estimators = 200, max_features=0.6, 
                                    max_depth=15).fit(df_train_data[train], df_train_target[train])
        pred=svc.predict(df_train_data[test])
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            score = pred_score(pred,df_train_target[test])
        print "predict score: {0:.3f}".format(score)
    '''

def modelTuningParams(X,y):
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
    print len(X_train),len(X_test)
    tuned_parameters = [{'n_estimators':[10,100,200,500]}]
    scores = ['r2'] #scoring评分器，通常设置为 r2
    for score in scores:
        print score
        model = GridSearchCV(RandomForestRegressor(),tuned_parameters,cv=5,scoring = score)#寻找最好的参数设置
        model.fit(X_train,y_train)

        #最优模型
        print '最优模型：'
        print (model.best_estimator_)
        print ('得分为：')
        pred = model.predict(X_test)
        for i in range(pred.shape[0]):
            #print pred[i],df_train_target[test]
            error = pred_score(pred,y_test)
        print "predict error: {0:.3f}".format(error)
'''
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)'''

#画学习曲线，观察是否过拟合
def plotLearningCurve(estimator,title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) 
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, #x,y
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    
    
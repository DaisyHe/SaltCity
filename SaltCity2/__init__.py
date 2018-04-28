# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 17:26:52 2018

@author: Administrator
"""

from preprocess import getTrainData,sepTargetAndOther,setMissinglevel_id,monthIs11
from preprocess import scaler,normalization,statistics,discrete,histgram
from preprocess import class_id_split,cluster,trend
from score import pred_score
from modelparameters import modelTuningParams,plotLearningCurve,testModel,testXGBoost,testGBDT
#from figure import drawPicTest
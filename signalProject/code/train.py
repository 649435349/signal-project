# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv

def dataGroup():
    '''
    读取数据
    '''
    arr=np.array(pd.read_csv('modifiedTrain.csv'))
    arrx=arr[:,:-1]
    arry=arr[:,-1].reshape((arr.shape[0],1))
    arrw=np.ones(150).reshape((150,1))
    return arrx,arry,arrw

def sigmoid(arr):
    '''
    sigmoid函数
    '''
    return 1/(1+np.exp(-arr))

def train(arrx,arry,arrw):
    '''
    梯度下降
    '''
    steps=100000
    alpha = 0.001
    for i in range(steps):
        t=(arry-sigmoid(arrx.dot(arrw))).reshape((arrx.shape[0]))
        arrw+=alpha*((t.dot(arrx)).reshape((arrw.shape[0],1)))
    return arrw

def write(arrw):
    '''
    参数输出
    '''
    with open('arrw.csv','w+') as f:
        writer=csv.writer(f)
        l=list(arrw.T[0])
        writer.writerow(l)

if __name__=='__main__':
    arrx,arry,arrw=dataGroup()
    arrw=train(arrx,arry,arrw)
    #write(arrw)

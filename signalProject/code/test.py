# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from train import sigmoid


def getData():
    arrw=np.array(pd.read_csv('arrw.csv',header=None)).reshape((150,1))
    arr=np.array(pd.read_csv('test.csv'))
    arrx = arr[:, :-1]
    arry = arr[:, -1].reshape((arr.shape[0], 1))
    return arrx,arry,arrw

def test(arrx,arry,arrw):
    t=sigmoid(arrx.dot(arrw))
    t=np.where(t>0.5,1,0)
    count=0
    for i in range(t.shape[0]):
        if t[i][0]==arry[i][0]:
            count+=1
    print '正确率为{}%'.format(count*100/t.shape[0])

if __name__=='__main__':
    arrx,arry,arrw=getData()
    test(arrx,arry,arrw)

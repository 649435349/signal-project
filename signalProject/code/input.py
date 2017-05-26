# -*- coding: utf-8 -*-
import csv
import pandas as pd
from scipy import signal
import numpy as np


def sample():
    '''
    这里用于截取200HZ里的150个采样点，包括了PQRS和U波群。
    时间是0.75秒，理论上都采样成功了。
    '''
    b,a=signal.butter(3,0.3,'low')
    t=pd.read_csv('sy200.csv',header=None)[0]
    #t=list(t[::5])#调频到200HZ
    sf=signal.filtfilt(b,a,t)#滤波信号
    sf=sf[13500:13650]
    sf=np.append(sf[sf.argmax():],sf[:sf.argmax()])
    return sf

def normalizationAndWritein(sf):
    with open('test.csv','a+') as f:#test.csv或者train.csv
        writer=csv.writer(f)
        sf=(sf-np.min(sf))/(np.max(sf)-np.min(sf))
        sf=np.append(sf,[0])#1或者0
        writer.writerow(sf)

if __name__=='__main__':
    sf=sample()
    normalizationAndWritein(sf)

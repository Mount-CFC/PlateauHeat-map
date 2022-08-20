
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:03:59 2021

@author: ltg

goodtraces的截取与台阶作图
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def cut_traces(filepath, DistGap = 0.1):
    '''
    
    return:
        datacut: 切分好的数据的字典。key：曲线标号；value：【距离，电导】的array
    '''
    data = pd.read_csv(filepath, header=None, sep='\t')
    data = np.array(data.iloc[:, :])
    LastPoint, data_column = data.shape
    
    DistDiff = np.diff(data[:, 0][:]) 
   
    TraceIndex = np.where(np.abs(DistDiff) > DistGap)[0] + 1 
    
    
    global all_traces
    all_traces = TraceIndex.shape[0] + 1   
    
    
    TraceIndex = TraceIndex.tolist()
    TraceIndex.append(LastPoint)  
    
    
    datacut = {}
    datacut[0] = data[:TraceIndex[0], :]
    for i in range(1, all_traces):
        datacut[i] = data[TraceIndex[i-1]:TraceIndex[i], :]
    
    
    print('CUT COMPLETE! Total goodtraces after cut:', len(datacut))
    return datacut

def cut_trace(data_after_cut, input_start, input_end):
    '''
    args:
        data_after_cut:切分好数据的字典
        intput_start:输入的低导
        input_end:输入的高导
        
    return：指定区间内每条曲线的台阶长度-》array
    '''
    start = []
    end = []

    for i in range(all_traces):

        
        cond = data_after_cut[i][:, 1]
        dist = data_after_cut[i][:, 0]
  
        temp_start = dist[cond >= input_start]   #电导值所有大于start的值，取最后一个
        temp_end = dist[cond <= input_end]   #电导值所有小于end的值，取第一个

        if temp_start.size and temp_start.size:
            dist_start = temp_start[-1]   
            dist_end = temp_end[0]
            start.append(dist_start)
            end.append(dist_end)
    global picked_number
    picked_number = len(start)            
    print('Total traces after pick:', picked_number)
    start = np.array(start)
    end = np.array(end)
    pleatu = start - end
    return pleatu


def draw_pleatu_all(pleatu):
   
    
    plt.hist(pleatu, range = (0, 3), bins=100, color='g')
    plt.show()





    
#%%
if __name__ == '__main__':
    filepath = r'Z:/Data/lutaige/data_processing/20210310_com4+highMA/com4+MA25-36.txt'
    #copy filepath of .txt
    data_after_cut = cut_traces(filepath)
    #%%
    
    end = -0.3      #pleatu start
    start = -3.6    #pleatu end
    

    pleatu = cut_trace(data_after_cut, start, end)
    draw_pleatu_all(pleatu)
    
 
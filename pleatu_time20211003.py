# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:03:59 2021

@author: ltg
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def cut_traces(filepath = r'', delta_conduct=4.0):
    '''
    这一步比较慢，如何提速？？（1w7条10min）
    return:
        datacut: 切分好的数据的字典。key：曲线标号；value：【距离，电导】的array
    '''
    # get start and end point of each trace
    
    dataset=np.loadtxt(filepath)
    delta_cond=delta_conduct
    datacut={}
    M1=len(dataset)
    pick_index=[0]
    for i in range(1,M1):
        temp = dataset[i,1] - dataset[i-1,1]
        if temp > delta_cond:
            pick_index.append(i-1)
            pick_index.append(i)
    pick_index.append(M1-1)
    
    N1=len(pick_index)
    count_i=0     #曲线编号
    #get the dictionary of all traces
    for i in range(0,N1,2):
        datacut[count_i]=dataset[pick_index[i]:pick_index[i+1]]
        count_i+=1
    global all_traces
    all_traces = len(datacut)
    print(f'CUT COMPLETE! Total goodtraces after cut:{all_traces}')
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

    # print(all_traces)
    for i in range(all_traces):

        # global cond
        # global dist
        # global single_trace
        # single_trace = data_after_cut[i]
        cond = data_after_cut[i][:, 1]
        dist = data_after_cut[i][:, 0]
  

        
        temp_start = dist[cond >= input_start]   #电导值所有大于start的值，取最后一个
        temp_end = dist[cond <= input_end]   #电导值所有小于end的值，取第一个

     
        if temp_start.size and temp_start.size:#判断不为空列表
            dist_start = temp_start[-1]   #防止起跳曲线的出现，取最后一个点
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
    # fig,(ax1,ax2) = plt.subplots(1,2,figsize=[9, 4], dpi=100)
    
    plt.hist(pleatu, range = (0, 3), bins=100, color='g')
    plt.show()


#%%
def get_axes_values(pleatu, 
           max_length, 
           bins_set, 
           trace_per_min,
           time_bin):
    '''获得画图所需的dataframe用于做透视表
    args:
        pleatu: 每条曲线的array
        max_length:台阶从0统计到哪里
        bins_set:histogram的bin值，越小方块越大
        traces_per_min：一分钟有多少条曲线
        time_bin = 时间，每隔多少时间统计一次pleatu的histogram
    
    return：
        index_y：时间点-》list
        bins：将台阶长度分隔成各个区间-》array
        value：每个时间的值-》array
    '''
    #调用hist函数，获得绘图所需的x轴length和“热力数值”
    
    traces_per_unit = time_bin * trace_per_min

    #先生成一个value的array，用于后续添加
    value, bins = np.histogram(pleatu[0 : 0+traces_per_unit], 
                               range = (0, max_length), 
                               bins = bins_set)
    index_y = [time_bin]
    
    for i in range(traces_per_unit, picked_number, traces_per_unit):#5 min
        pleatu_per_unit = pleatu[i : i+traces_per_unit]

        hist1, bin_edges = np.histogram(pleatu_per_unit, 
                                        range = (0, max_length), 
                                        bins = bins_set)
  
        value = np.vstack((value, hist1))
        
        index_y.append(time_bin + (i//traces_per_unit)*time_bin)
    value = value[0 : len(value)-1] #删除最后一个因为余数而增加的array
    index_y.pop() #删除最后一个因为余数而增加的array的index
    
   
   
    bins = np.around(bins, decimals=2)
    return index_y, bins, value, traces_per_unit#后面gauss拟合用
#%%
def get_dataframe(start_time, end_time):
    '''生成一个格式为dataframe的透视表
    可以根据开始、结束时间(start_time, end_time)进行index/value的切片
    '''
    
    start_index = start_time//time_bin #开始的编号，用于y轴、value的定位
    end_index = end_time//time_bin
    
    
    pivot_df = pd.DataFrame(value[start_index: end_index],
                      columns = bins[0:bins_set],
                      )
    
    pivot_df['time'] = index_y[start_index : end_index]

    return pivot_df, start_index, end_index #后面gauss拟合用

#%%    
def gauss_fit(start_index, end_index, traces_per_unit):
    # fit_max = []
    fit_avg = []
    for i in range(start_index, end_index):
        miu = np.mean(pleatu[i*traces_per_unit : (i+1)*traces_per_unit])
        # maxima = np.max(pivot_dataframe.iloc[i][:-1])
        # fit_max.append(maxima)
        fit_avg.append(miu)
    
    # print(fit_avg, fit_max)
    y = index_y[start_index : end_index]
    # x_max = fit_max
    x_avg = fit_avg
    plt.figure(figsize=(15, 5), dpi=100)
    plt.plot(x_avg, y,color = 'b', 
              
             linewidth = 3, 
             marker = '*', 
             markersize = 20
             )
    
    plt.xlim(0, max_length)
    plt.gca().invert_yaxis()
    
    plt.xlabel('Arvage length of a time unit')
    plt.ylabel('Time(min)')
    
    plt.show()
    
    

    
#%%
if __name__ == '__main__':
    filepath = r'MPSlongtimecontrol.txt'
    data_after_cut = cut_traces(filepath)
    #%%
    
    end = -0.3
    start = -3.7
    
    bins_set = 20
    pleatu = cut_trace(data_after_cut, start, end)
    
    minutes = 228   #生成的这些曲线共用时多久
    
    trace_per_min = picked_number//minutes #每分钟都少条曲线
    max_length = 0.6 #台阶从0统计到哪里
    bins_set = 60    #histogram的bin值，越小方块越大
    time_bin = 5  #时间，每隔多少时间统计一次台阶长度，值决定了方块纵向大小
    
    '''设置绘图坐标'''
    start_time = 5 #画图时开始的时间坐标,默认为0
    # start_time = 30
    # end_time=minutes   #画图时结束的时间坐标，默认到结束
    end_time = 200
    
    # draw_pleatu_all(pleatu)
    

    
    
    index_y, bins, value, traces_per_unit = get_axes_values(pleatu, 
           max_length, 
           bins_set, 
           trace_per_min,
           time_bin)
   
    
    
    
    pivot_dataframe, start_index, end_index = get_dataframe(start_time, end_time)
    
    
    
    
    #生成逆透视后的dataframe
   
    melted_df = pivot_dataframe.melt(id_vars = 'time',
                              value_vars = bins[0:bins_set])

    target = pd.pivot_table(data = melted_df, values = 'value', 
                        index = 'time', columns = 'variable')
   
    ax = plt.figure(figsize = (15, 5),#(底，高)
                    dpi = 100
                    )
    # 画热力图
   
    ax = sns.heatmap(target, # 指定绘图数据                 
                     # cmap=plt.cm.Blues, # 指定填充色 
                     cmap = 'coolwarm',
                     linewidths=0, # 设置每个单元方块的间隔  为0               
                     annot=False, # True时显示数值
                     #调整colarbar
                     # vmax：设置颜色带的最大值，vmin：设置颜色带的最小值，center：设置颜色带的分界线
                      #  vmax=30,
                       # center=8,
                       #  vmin=1
                     )
    
    
    ax.set_title(f'Total traces:{picked_number} (during {minutes} min); max length:{max_length} nm; bins set:{bins_set}\nlogG range:{start} ~ {end}; Time range:{start_time}-{end_time} min; time bin:{time_bin} min')
    ax.set_xlabel(f'Pleatu length(nm)\nFilepath:{filepath}')
    ax.set_ylabel('Time(min)')
    plt.rcParams['font.sans-serif'] = 'SimHei'#设置中文显示
    plt.rcParams['axes.unicode_minus'] = False #更改中文字体后，会导致坐标轴中的部分字符无法显示，同时要更改axes.unicode_minus参数
    
  
    
    plt.show()
    

  
    gauss_fit(start_index, end_index, traces_per_unit)
    print('drawing complete!')
    

    
   

    
    

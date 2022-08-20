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


def cut_traces(filepath, DistGap = 0.1):
    '''
    
    return:
        datacut: 切分好的数据的字典。key：曲线标号；value：【距离，电导】的array
    '''
    data = pd.read_csv(filepath, header=None, sep='\t')
    data = np.array(data.iloc[:, :])
    LastPoint, data_column = data.shape
    
    DistDiff = np.diff(data[:, 0][:]) 
    # np.where(np.abs(DistDiff) > distanceGap) 返回的是元组。需要[0]索引
    TraceIndex = np.where(np.abs(DistDiff) > DistGap)[0] + 1 #每一条曲线的开始，除了第一条
    #distance gap == 0.1识别不同曲线
    
    global all_traces
    all_traces = TraceIndex.shape[0] + 1 #加第一条   
    # print(f'CUT COMPLETE! Total goodtraces after cut:{all_traces}')
    
    
    TraceIndex = TraceIndex.tolist()
    TraceIndex.append(LastPoint)  
    #加上最后一条曲线的最后一个元素,为第2条到最后一条曲线的起点和最后一条曲线的最后一个点
    
    datacut = {}
    datacut[0] = data[:TraceIndex[0], :]  #先加上第一条曲线的距离，电导数据
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

    # print(all_traces)
    for i in range(all_traces):

        # global cond
        # global dist
        # global single_trace
        # single_trace = data_after_cut[i]
        cond = data_after_cut[i][:, 1]
        dist = data_after_cut[i][:, 0]
  

        # https://blog.csdn.net/u012193416/article/details/79672514
        # 如果是1维数组，返回一维的列表，直接是相应元素编号 的一个列表 eg.（【1】，【2】）
        # 参考：
        # x = np.arange(5)
        # print(x, '\n', x>2, '\n',x[x>2], '\n',x[x>2][0], sep = '')
        # global temp_start, temp_end, dist_start, dist_end
        temp_start = dist[cond >= input_start]   #电导值所有大于start的值，取最后一个
        temp_end = dist[cond <= input_end]   #电导值所有小于end的值，取第一个

        # https://blog.csdn.net/u011622208/article/details/103683358?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
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
        # https://blog.csdn.net/A_pinkpig/article/details/105333946
        # https://blog.csdn.net/lvbu89757/article/details/97891822
        # https://blog.csdn.net/weixin_45860697/article/details/103270244
        # https://blog.csdn.net/csdn15698845876/article/details/73380803
        value = np.vstack((value, hist1))
        
        index_y.append(time_bin + (i//traces_per_unit)*time_bin)
    value = value[0 : len(value)-1] #删除最后一个因为余数而增加的array
    index_y.pop() #删除最后一个因为余数而增加的array的index
    
    # https://blog.csdn.net/lixiaowang_327/article/details/82149628 
    # 小数点后保留两位，为了画图美观
    bins = np.around(bins, decimals=2)
    return index_y, bins, value, traces_per_unit #后面gauss拟合用
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
    # https://blog.csdn.net/zx1245773445/article/details/99445332 新增一列
    pivot_df['time'] = index_y[start_index : end_index]

    return pivot_df, start_index, end_index #后面gauss拟合用

#%%    
def gauss_fit(start_index, end_index, traces_per_unit):
    fit_max = []
    fit_avg = []
    for i in range(start_index, end_index):
        miu = np.mean(pleatu[i*traces_per_unit : (i+1)*traces_per_unit])
        temp = bins[0:bins_set]
        maxima = np.mean(temp[value[i] == np.max(pivot_dataframe.iloc[i][:-1])])
        fit_max.append(maxima)
        fit_avg.append(miu)
    
    # print(fit_avg, fit_max)
    y = index_y[start_index : end_index]
    x_max = fit_max
    x_avg = fit_avg
    
    F, ax = plt.subplots(2, 1, figsize = (15, 10), dpi = 100,
                             )
    ax[0].plot(x_avg, y, color = 'b',linewidth = 4, marker = '*', markersize = 20)
    ax[1].plot(x_max, y, color = 'r', linewidth = 4, marker = 'D', markersize = 20)
    ax[0].set_xlabel('Arvage length of a time unit',fontsize=18)
    ax[1].set_xlabel('Max length of a time unit',fontsize=18)
    ax[0].set_ylabel('Time(min)' ,fontsize=18)
    ax[1].set_ylabel('Time(min)',fontsize=18)
    ax[0].set_xlim(0,max_length)
    ax[1].set_xlim(0,max_length)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    
    #单独显示使用下列代码：
    # plt.figure(figsize=(15, 5), dpi=100)
    # plt.plot(x_avg, y,color = 'b', 
              
    #          linewidth = 3, 
    #          marker = '*', 
    #          markersize = 20
    #          )
    
    # plt.xlim(0, max_length)
    # plt.gca().invert_yaxis()
    
    # plt.xlabel('Arvage length of a time unit')
    # plt.ylabel('Time(min)')
    
    # plt.show()
    
    
    # plt.figure(figsize=(15, 5), dpi=100)
    # plt.plot(x_max, y,color = 'r', 
              
    #           linewidth = 3, 
    #           marker = '*', 
    #           markersize = 20
    #           )
    
    # plt.xlim(0, max_length)
    # plt.gca().invert_yaxis()
    
    
    # plt.xlabel('Max length of a time unit')
    # plt.ylabel('Time(min)')
    
    # plt.show()

    
#%%
if __name__ == '__main__':
    filepath = r'X:\Data\lutaige\data_processing\20210824MenshutinTCB\MenshutkinTCB.txt'
    data_after_cut = cut_traces(filepath)
    #%%
    
    end = -0.3
    start = -3.6
    

    pleatu = cut_trace(data_after_cut, start, end)
    
    minutes = 200   #生成的这些曲线共用时多久
    
    trace_per_min = picked_number//minutes #每分钟都少条曲线
    max_length = 0.6 #台阶从0统计到哪里
    bins_set = 30    #histogram的bin值，越小方块越大
    time_bin =2  #时间，每隔多少时间统计一次台阶长度，值决定了方块纵向大小
    
    '''设置绘图坐标'''
    start_time = 0 #画图时开始的时间坐标,默认为0
    # start_time = 30
    # end_time=minutes   #画图时结束的时间坐标，默认到结束
    end_time = 30
    
    # draw_pleatu_all(pleatu)
    

    
    
    index_y, bins, value, traces_per_unit = get_axes_values(pleatu, 
           max_length, 
           bins_set, 
           trace_per_min,
           time_bin)
   
    
    
    
    pivot_dataframe, start_index, end_index = get_dataframe(start_time, end_time)
    
    
    
    
    #生成逆透视后的dataframe
    # https://www.cnblogs.com/ljhdo/p/11591958.html 逆透视
    melted_df = pivot_dataframe.melt(id_vars = 'time',
                              value_vars = bins[0:bins_set])
    # http://www.360doc.com/content/17/1014/10/42030643_694805456.shtml
    target = pd.pivot_table(data = melted_df, values = 'value', 
                        index = 'time', columns = 'variable')
   
    ax = plt.figure(figsize = (10, 5),#(底，高)
                    dpi = 100
                    )
    # 画热力图
    # https://blog.csdn.net/weixin_39934085/article/details/111293624?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-9.no_search_link&spm=1001.2101.3001.4242
    ax = sns.heatmap(target, # 指定绘图数据                 
                     # cmap=plt.cm.Blues, # 指定填充色 
                     cmap = 'coolwarm',
                     linewidths=0, # 设置每个单元方块的间隔  为0               
                     annot=False, # True时显示数值
                     #调整colarbar
                      # vmax：设置颜色带的最大值，vmin：设置颜色带的最小值，center：设置颜色带的分界线
                        vmax=20,
                        center=8,
                         # vmin=1
                     )
    

    
    
    ax.set_title(f'Filepath:{filepath}\nTotal traces:{picked_number} (during {minutes} min); max length:{max_length} nm; bins set:{bins_set}\nlogG range:{start}, {end}; Time range:{start_time}-{end_time} min; time bin:{time_bin} min')
    ax.set_xlabel('Pleatu length(nm)',fontsize=12)
    ax.set_ylabel('Time(min)',fontsize=10)
    plt.rcParams['font.sans-serif'] = 'SimHei'#设置中文显示
    plt.rcParams['axes.unicode_minus'] = False #更改中文字体后，会导致坐标轴中的部分字符无法显示，同时要更改axes.unicode_minus参数
    
    '''这里修改x轴的标签
    https://blog.csdn.net/Poul_henry/article/details/82590392?utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-8.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-8.no_search_link
    #plt.xticks(np.arange(bins_set)+0.5, labels = bins[0:bins_set])
    debug：并不需要了，将修改小数点round()后的bins赋值给bins即可。
    '''
    
    

  #%%
    # gauss_fit(start_index, end_index, traces_per_unit)
    print('drawing complete!')
    fit_max = []
    fit_avg = []
    for i in range(start_index, end_index):
        miu = np.mean(pleatu[i*traces_per_unit : (i+1)*traces_per_unit])
        temp = bins[0:bins_set]
        maxima = np.mean(temp[value[i] == np.max(pivot_dataframe.iloc[i][:-1])])
        fit_max.append(maxima)
        fit_avg.append(miu)
    
    
    # # print(fit_avg, fit_max)
    y = index_y[start_index : end_index]
    x_max = np.array(fit_max)
    x_avg = np.array(fit_avg)
    # ax.plot(x_avg, y, color = 'b',linewidth = 4, marker = '*', markersize = 20)
    ax.plot(x_max*30 ,y, color = 'b',linewidth = 4, marker = '*', markersize = 20)
    plt.show()
    # F, ax = plt.subplots(2, 1, figsize = (15, 10), dpi = 100,
    #                          )
    # ax[0].plot(x_avg, y, color = 'b',linewidth = 4, marker = '*', markersize = 20)
    # ax[1].plot(x_max, y, color = 'r', linewidth = 4, marker = 'D', markersize = 20)
    # ax[0].set_xlabel('Arvage length of a time unit',fontsize=18)
    # ax[1].set_xlabel('Max length of a time unit',fontsize=18)
    # ax[0].set_ylabel('Time(min)' ,fontsize=18)
    # ax[1].set_ylabel('Time(min)',fontsize=18)
    # ax[0].set_xlim(0,max_length)
    # ax[1].set_xlim(0,max_length)
    # ax[0].invert_yaxis()
    # ax[1].invert_yaxis()
    

    
   

    
    

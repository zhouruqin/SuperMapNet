import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

# 加载 Times New Roman 字体
#font = FontProperties(family='Times New Roman', size=12)
# 数据

plt.rc('font',family='Times New Roman')

methods_c = ["BeMapNet-30",  "MapQR-110", "MapTR-110", "MapTRv2-110", "PivotNet-110","HIMapNet-110" ]
epoch_values_c = [4.2,   11.9, 15.1, 14.1, 9.2, 11.4, ]
map_values_c = [ 62.4, 72.6, 59.3, 68.7, 66.8, 73.7, ]

methods_cl = [ "MapTR-24", "MapTRv2-24", 
           "GeMap-110", "ADMap-110", 
           "Ours-30"]
epoch_values_cl = [  6.0, 5.8 , 
                   6.8, 5.8, 5.0]
map_values_cl = [ 62.5, 69.0, 
                70.4, 71.5, 86.8]
'''
methods1 = [ "MapTR-24", "MapTRv2-24", 
            "BeMapNet-30", 
           "MapQR-110", "MapTR-110", "MapTRv2-110", "PivotNet-110","HIMapNet-110", 
           "GeMap-110", "ADMap-110", 
           "Ours-30"]
epoch_values1 = [  6.0, 5.8 , 
                  4.2, 11.4, 
                 11.9, 15.1, 14.1, 9.2, 11.4, 6.8, 5.8, 5.0]
map_values1 = [ 62.5, 69.0, 
                62.4,  
               72.6, 59.3, 68.7, 66.8, 73.7, 70.4, 71.5, 86.8]
'''
'''
methods1 = ['MapTR-6', 'MapTRv2-6', "MapVR-6", "HIMapNet-6",
           "MapQR-6", "InsMapper-6", "HDMapNet-6","ADMapNet-6",
           "VectorMapNet-24", "GeMap-24", "Ours-6"]

epoch_values1 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
map_values1 = [56.5, 67.5, 57.5, 69.6, 68.2, 61.6,18.8, 75.2, 37.9, 71.8, 86.9]
'''
# 创建散点图
plt.figure(figsize=(8,6))

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用黑体（Windows 系统）
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux 系统

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  

colors = plt.cm.rainbow(np.linspace(0, 1, len(methods_c)+len(methods_cl)))
markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'H',  'D', 'd',]

custom_handles = [
    Line2D([0], [0], 
           marker='*', 
           color='b', 
           linestyle='',
           markersize=10,
           label='Camera only'),
    Line2D([0], [0], 
           marker='v', 
           color='r',
           linestyle='',
           markersize=10,
           label='Camera+LiDAR')
]


for i, method in enumerate(methods_c):
    marker = markers[i % len(markers)]
    plt.scatter(epoch_values_c[i],map_values_c[i],  color='b', marker='*',  s=80)
    plt.text(epoch_values_c[i],map_values_c[i], method, fontsize=10, ha='center' , va = 'bottom')
    
for i, method in enumerate(methods_cl):
    marker = markers[(i+len(methods_c)) % len(markers)]
    plt.scatter(epoch_values_cl[i],map_values_cl[i],  color='r', marker='v',  s=80)
    plt.text(epoch_values_cl[i],map_values_cl[i], method, fontsize=10, ha='center' ,  va = 'bottom')

# 设置横轴刻度间隔为30
#plt.xticks(np.arange(0, 18, 1))#nu
#plt.yticks(np.arange(30, 90, 10))
plt.yticks(np.arange(60, 89, 3), fontproperties='Times New Roman',)#arg2np.arange(15, 90, 10)
plt.xticks(np.arange(3, 17, 1), fontproperties='Times New Roman',)#np.arange(0, 16, 1)
# 添加图例
#plt.legend()
plt.legend(handles=custom_handles,  loc='upper right',
           fontsize=10,
           framealpha=0.9)

# 添加标签和标题
plt.xlabel('FPS',fontsize=12, )
plt.ylabel('mAP',fontsize=12,  )
#plt.title('mAP of different methods on Argoverse2 dataset',fontsize=15  )
plt.title('mAP VS FPS on nuScenes dataset',fontsize=12  )

# 保存图片
plt.savefig('mAP_vs_FPS_nu.png', dpi=300, bbox_inches='tight')

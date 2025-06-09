import numpy as np
import matplotlib.pyplot as plt
import os


colors_plt = ['r', 'b', 'g']


def visual_map_gt_pivot(map_region, vectors, i, root_name):     #range is [map_region]
    filename = root_name + 'gt_map_pivot'
    img_name = filename + '/' +  str(i) + '_gt.svg'#jpg
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    #if os.path.exists(img_name):
    plt.figure(figsize=((map_region[1]-map_region[0]), map_region[3]-map_region[2]))
    plt.xlim(map_region[0], map_region[1])
    plt.ylim(map_region[2], map_region[3])
    plt.axis('off')

    ax = plt.axes()
    for line_type, pts in vectors['points'].items():
        for curr_pts in pts:
            if curr_pts is not None:
                #pts, pts_num, line_type = vector['points'], vector['valid_len'], vector['masks']
                x_start, y_start =  curr_pts[0, 0], curr_pts[0, 1]
                #print(pts.shape, pts)
                for k in range(1, len(curr_pts)):  
                    x_end, y_end = curr_pts[k, 0], curr_pts[k, 1]
                    if x_end==0 and y_end==0:
                        break
                    else:
                        #print(x_start, y_start, x_end, y_end)
                        # color = colors[idx % len(colors)]
                        plt.scatter(x_start, y_start, c='black', s=500)
                        plt.scatter(x_end, y_end, c='black', s=500)
                        ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color=colors_plt[line_type], head_width=0,linewidth=10)
                        x_start, y_start = x_end, y_end
                #x = np.array([pt[0] for pt in pts])
                #y = np.array([pt[1] for pt in pts])
                #plt.plot(x, y, color=colors_plt[line_type])

    #print('saving', img_name)
    plt.axis('off')
    plt.savefig(img_name, dpi=400,format="svg")
    plt.close()
    #else:
    #    print('No such file!')    
    
    
def visual_map_gt(map_region, vectors, i, root_name):     #range is [map_region]
    filename = root_name + 'gt_map'
    img_name = filename + '/' +  str(i) + '_gt.svg'#jpg
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    plt.figure(figsize=((map_region[1]-map_region[0]), map_region[3]-map_region[2]))
    plt.xlim(map_region[0], map_region[1])
    plt.ylim(map_region[2], map_region[3])
    plt.axis('off')

    ax = plt.axes()
    for  idx,  vector in enumerate(vectors):
        if vector is not None:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = np.array(pts[:pts_num])#.cpu().detach().numpy()
            # pts = pts[0, :]
            #print('1', pts)
            x_start, y_start = pts[:, ::-1][:, 0][0], pts[:, ::-1][:, 1][0]
            for k in range(1, len(pts)):
                x_end, y_end = pts[:, ::-1][:, 0][k],pts[:, ::-1][:, 1][k]
                # color = colors[idx % len(colors)]
                plt.scatter(x_start, y_start, c='black', s=500)
                plt.scatter(x_end, y_end, c='black', s=500)
                ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color=colors_plt[line_type], head_width=0,linewidth=10)
                x_start, y_start = x_end, y_end
            #x = np.array([pt[0] for pt in pts])
            #y = np.array([pt[1] for pt in pts])
            #plt.plot(x, y, color=colors_plt[line_type])

    #print('saving', img_name)
    plt.axis('off')
    plt.savefig(img_name, dpi=400,format="svg")
    plt.close()
    #else:
    #    print('No such file!')    
        
def visual_map_pred(map_region, dt_res, i, root_name):  #[800, 200]
    filename = root_name + 'pred_map_bev_cat'
    img_name = filename + '/' +  str(i) + '_pred.svg'#jpg
    if not os.path.exists(filename):
        os.makedirs(filename)

    #if os.path.exists(img_name):
    plt.figure(figsize=((map_region[1]-map_region[0]), map_region[3]-map_region[2]))
    plt.xlim(map_region[0], map_region[1])
    plt.ylim(map_region[2], map_region[3])
    plt.axis('off')

    ax = plt.axes()
    
    dt_res = dt_res#.tolist()
    line_type = dt_res['pred_label']
    #del line_type[0]
    line_type = [x - 1 for x in line_type]
    #del coords[0]
    #print(dt_res['map'])
    for idx, pts in enumerate(dt_res['map']):
        if pts is not None:
            #print('0', pts)
            x_start, y_start = pts[:, 0][0], pts[:, 1][0] # 第一个箭头的起点
            for k in range(1, len(pts)):
                x_end, y_end = pts[:, 0][k], pts[:, 1][k]
                plt.scatter(x_start, y_start, c='black', s=500)
                plt.scatter(x_end, y_end, c='black', s=500)
                ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color=colors_plt[line_type[idx]], head_width=0,linewidth=10)
                x_start, y_start = x_end, y_end
                    
    #print('saving', img_name)
    plt.axis('off')
    plt.savefig(img_name, dpi=400,format="svg")
    plt.close() 
    #else:
    #    print('No such file!')      


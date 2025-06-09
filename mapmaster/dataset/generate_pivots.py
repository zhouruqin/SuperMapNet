import numpy as np
import visvalingamwyatt as vw



def calculate_bumps(points, min_bump_length):
    # 计算相邻点之间的差异
    differences = np.diff(points, axis=0)
    angle = differences[:, 1]/differences[:, 0]
    print(angle)
    angle[np.abs(angle) < 1] = 0
    # 计算差异的符号变化
    #sign_changes = np.diff(np.signbit(angle), axis=0)
    # 识别凸起的开始和结束
    bumps = np.where(angle != 0)[0]
    
    # 计算凸起长度和凸起点集
    bump_lengths = []
    filtered_points = []
    current_index = 0
    
    print(bumps)
    for i in range(bumps.shape[0]-1):  
        bump_start = bumps[i]
        bump_end = bumps[i+1]
        dis = (points[bump_end][0] - points[bump_start][0])**2+(points[bump_end][1] - points[bump_start][1])**2
        #print(dis)
        if dis>0.2 and dis<min_bump_length:  # 凸起长度大于1
            bump_lengths.append(dis)
            filtered_points.append(points[current_index:bump_start])
        current_index = bump_end +1
    
    filtered_points.append(points[current_index:])
    
    return np.concatenate(filtered_points)


class GenPivots:
    def __init__(self, max_pts, map_region, vm_thre=2.0, resolution=0.15):
        self.max_pts = max_pts
        self.map_region = map_region
        self.vm_thre = vm_thre
        self.resolution = resolution
        
    def pivots_generate(self, map_vectors):
        pivots_single_frame =  {0:[], 1:[], 2:[]}
        lengths_single_frame =  {0:[], 1:[], 2:[]}
        for ii, vec in enumerate(map_vectors):
            pts = np.array(vec["pts"]) * self.resolution  # 转成 m
            pts = pts[:, ::-1]
            cls = vec["type"]
        
            # If the difference in x is obvious (greater than 1m), then rank according to x. 
            # If the difference in x is not obvious, rank according to y.
            if (np.abs(pts[0][0]-pts[-1][0])>1 and pts[0][0]<pts[-1][0]) \
                or (np.abs(pts[0][0]-pts[-1][0])<=1 and pts[0][1]<pts[-1][1]): 
                pts = pts[::-1]
        
            simplifier = vw.Simplifier(pts)
            sim_pts = simplifier.simplify(threshold=self.vm_thre)
            #print('0', sim_pts.shape)
            #if len(sim_pts)>4:
            #    sim_pts = calculate_bumps(points=sim_pts, min_bump_length=400)#av2 data only  avw数据集标记中有很多凸起
            #    print('1', sim_pts.shape)
            length = min(self.max_pts[cls], len(sim_pts))

            padded_pts = self.pad_pts(sim_pts, self.max_pts[cls])
            pivots_single_frame[cls].append(padded_pts)
            lengths_single_frame[cls].append(length)
            
        
        

        for cls in [0, 1, 2]:
            new_pts = np.array(pivots_single_frame[cls])
            if new_pts.size > 0:
                new_pts[:, :, 0] = new_pts[:, :, 0] / (self.map_region[1]- self.map_region[0])  # normalize
                new_pts[:, :, 1] = new_pts[:, :, 1] / (self.map_region[3] - self.map_region[2])
            pivots_single_frame[cls] = new_pts
            lengths_single_frame[cls] = np.array(lengths_single_frame[cls])
            
        return pivots_single_frame, lengths_single_frame
    
    def pad_pts(self, pts, tgt_length):
        if len(pts) >= tgt_length:
            return pts[:tgt_length]
        pts = np.concatenate([pts, np.zeros((tgt_length-len(pts), 2))], axis=0)
        return pts

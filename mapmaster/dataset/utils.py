import numpy as np
import torch
from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import ops
import numpy as np
from scipy.spatial import distance
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from shapely.affinity import translate

def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1) 
    return onehot


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) 
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) 
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.
        
    Args:
        geom (BaseGeometry): geoms to be split or validate.
    
    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom,]
        else:
            return []


def get_drivable_area_contour(drivable_areas: List[Polygon], 
                              roi_size: Tuple) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
        roi_size (tuple): bev range size
    
    Returns:
        boundaries (List): list of boundaries.
    '''
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # a bit smaller than roi to avoid unexpected boundaries on edges
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    exteriors = []
    interiors = []
    
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    
    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way
        
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    return results


def get_ped_crossing_contour(polygon: Polygon, 
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params
    
    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        
        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line) # make ped cross closed

    if not lines.is_empty:
        return lines
    
    return None


def remove_repeated_lines(lines: List[LineString]) -> List[LineString]:
    ''' Remove repeated dividers since each divider in argoverse2 is mentioned twice
    by both left lane and right lane.

    Args:
        lines (List): list of dividers

    Returns:
        lines (List): list of left dividers
    '''

    new_lines = []
    for line in lines:
        repeated = False
        for l in new_lines:
            length = min(line.length, l.length)
            #print(line.length, l.length)
            
            # hand-crafted rule to check overlap
            if line.buffer(0.5).intersection(l.buffer(0.5)).area > 0.5*length:
                repeated = True
                break
        
        if not repeated:
            new_lines.append(line)
    
    return new_lines


def remove_boundary_dividers(dividers: List[LineString], 
                             boundaries: List[LineString]) -> List[LineString]:
    ''' Some dividers overlaps with boundaries in argoverse2 dataset so
    we need to remove these dividers.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''
    #print(len(dividers))
    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        
        for bound in boundaries:
            length = min(divider.length, bound.length)
            #print(length)

            # hand-crafted rule to check overlap
            if divider.buffer(1.5).intersection(bound.buffer(1.5)).area \
                    > 1:
                # the divider overlaps boundary
                dividers.pop(idx)
                #print(divider.buffer(0.5).intersection(bound.buffer(0.5)).area,  0.2 * length, len(dividers))
                break

    return dividers


def line_rect_buffer(line, dis):
    # 向上平移5米
    translated_up = translate(line, xoff=0, yoff=dis)  # 向上平移5米
    translated_down = translate(line, xoff=0, yoff=-dis)  # 向下平移5米
  
    # 获取平移后的线段的坐标
    coords_up = list(translated_up.coords)
    coords_down = list(translated_down.coords)

    # 构造矩形缓冲区的顶点
    # 上矩形
    rect_up = Polygon([
        coords_up[0],  # 起点
        coords_up[-1],  # 终点
        coords_down[-1],  # 终点向下1米
        coords_down[0],  # 起点向下1米
    ])

    # 绘制向上平移后的矩形缓冲区
    return rect_up

def remove_ped_dividers(dividers: List[LineString], 
                             peds: List[LineString]) -> List[LineString]:
    ''' Some dividers overlaps with ped in argoverse2 dataset so
    we need to remove these dividers.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''
    #print(len(dividers))
    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        #divider_polygon = line_rect_buffer(divider)
        
        for ped in peds:
            #print(ped)
            length = min(divider.length, ped.length)
            #print(length)

            if ped.geom_type == 'LineString':
                ped_polygon = line_rect_buffer(ped, 20)
                intersection = divider.intersection(ped_polygon)
                total_length = intersection.length
            elif ped.geom_type == 'MultiLineString':
                total_length = 0
                for single_line in ped.geoms:
                    ped_polygon = line_rect_buffer(single_line, 20)
                    intersection = divider.intersection(ped_polygon)
                    total_length = max(total_length, intersection.length)
            else:
                total_length = 0
            # hand-crafted rule to check overlap
            if total_length > 0:
                # the divider overlaps boundary
                dividers.pop(idx)
                #print(divider.buffer(0.5).intersection(bound.buffer(0.5)).area,  0.2 * length, len(dividers))
                break

    return dividers


def connect_lines(lines: List[LineString]) -> List[LineString]:
    ''' Some dividers are split into multiple small parts
    so we need to connect these lines.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    new_lines = []
    eps = 0.1 # threshold to identify continuous lines
    while len(lines) > 1:
        line1 = lines[0]
        merged_flag = False
        for i, line2 in enumerate(lines[1:]):
            # hand-crafted rule
            begin1 = list(line1.coords)[0]
            end1 = list(line1.coords)[-1]
            begin2 = list(line2.coords)[0]
            end2 = list(line2.coords)[-1]

            dist_matrix = distance.cdist([begin1, end1], [begin2, end2])
            if dist_matrix[0, 0] < eps:
                coords = list(line2.coords)[::-1] + list(line1.coords)
            elif dist_matrix[0, 1] < eps:
                coords = list(line2.coords) + list(line1.coords)
            elif dist_matrix[1, 0] < eps:
                coords = list(line1.coords) + list(line2.coords)
            elif dist_matrix[1, 1] < eps:
                coords = list(line1.coords) + list(line2.coords)[::-1]
            else: continue

            new_line = LineString(coords)
            lines.pop(i + 1)
            lines[0] = new_line
            merged_flag = True
            break
        
        if merged_flag: continue

        new_lines.append(line1)
        lines.pop(0)

    if len(lines) == 1:
        new_lines.append(lines[0])

    return new_lines


def transform_from(xyz: NDArray, 
                   translation: NDArray, 
                   rotation: NDArray) -> NDArray:
    ''' Transform points between different coordinate system.

    Args:
        xyz (array): original point coordinates
        translation (array): translation
        rotation (array): rotation matrix

    Returns:
        left_dividers (list): list of left dividers
    '''
    
    new_xyz = xyz @ rotation.T + translation
    return new_xyz

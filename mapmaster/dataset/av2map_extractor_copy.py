from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import LineString, box, Polygon, MultiLineString
from shapely import ops, affinity
import numpy as np
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour, remove_repeated_lines, transform_from, \
        connect_lines, remove_boundary_dividers,remove_ped_dividers
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union
#import geopandas as gpd
import pandas as pd
import mmcv
import os
from av2.geometry.utm import convert_city_coords_to_utm, CityName
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from av2.geometry.se3 import SE3
from os import path as osp
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
import networkx as nx
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from av2.map.map_primitives import Polyline

CityNames = [
    "ATX",  # Austin, Texas
    "DTW",  # Detroit, Michigan
    "MIA",  # Miami, Florida
    "PAO",  # Palo Alto, California
    "PIT",  # Pittsburgh, PA
    "WDC"]

def get_patch_coord(patch_box: Tuple[float, float, float, float],
                    patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


class VectorizedAV2LocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'divider': 0,
        'ped_crossing': 1,
        'boundary': 2,
        'others': -1
    }

    def __init__(self,
                 dataroot,
                 patch_size,
                 test_mode=False,
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000, ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        # self.data_root = dataroot
        self.test_mode = test_mode
        #if self.test_mode:
        self.data_root = osp.join(dataroot, self.test_mode)
        

        self.loader = AV2SensorDataLoader(data_dir=Path(dataroot), labels_dir=Path(dataroot))

        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, map_elements, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        av2 lidar2global the same as ego2global
        location the same as log_id
        '''
        # avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion._from_matrix(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        #print(patch_box)
        vectors = []
        city_SE2_ego = SE3(lidar2global_rotation, lidar2global_translation)
        ego_SE3_city = city_SE2_ego.inverse()
       
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_divider_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                line_instances_list = self.line_geoms_to_instances(line_geom)
                for divider, length in line_instances_list:
                    vectors.append((divider, length, self.CLASS2LABEL.get('divider', -1)))#
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_ped_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                ped_instance_list = self.line_geoms_to_instances(ped_geom)
                for instance, length in ped_instance_list:
                    vectors.append((instance, length, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_boundary_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                poly_bound_list = self.bound_poly_geoms_to_instances(polygon_geom)
                for bound, length in poly_bound_list:
                    vectors.append((bound, length, self.CLASS2LABEL.get('boundary', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

           
        filtered_vectors = []
        
        for pts, pts_num, _type in vectors:
            if _type != -1:
                filtered_vectors.append({'pts': pts, 'pts_num': pts_num, 'type': _type})

        return filtered_vectors
    

    def proc_polygon(self, polygon, ego_SE3_city):
        # import pdb;pdb.set_trace()
        interiors = []
        exterior_cityframe = np.array(list(polygon.exterior.coords))
        exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
        for inter in polygon.interiors:
            inter_cityframe = np.array(list(inter.coords))
            inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
            interiors.append(inter_egoframe[:, :2])

        new_polygon = Polygon(exterior_egoframe[:, :2], interiors)
        return new_polygon

    def get_map_boundary_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        map_boundary_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        # import pdb;pdb.set_trace()
        polygon_list = []
        for da in avm:
            exterior_coords = da
            # import pdb;pdb.set_trace()
            interiors = []
            # import pdb;pdb.set_trace()
            is_polygon = np.array_equal(exterior_coords[0], exterior_coords[-1])
            if is_polygon:
                polygon = Polygon(exterior_coords, interiors)
            else:
                import pdb;
                pdb.set_trace()
                polygon = LineString(exterior_coords)
                raise ValueError(f'WRONG type: line in boundary')
            if is_polygon:
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        # import pdb;pdb.set_trace()
                        if new_polygon.geom_type is 'Polygon':
                            if not new_polygon.is_valid:
                                continue
                            new_polygon = self.proc_polygon(new_polygon, ego_SE3_city)
                            if not new_polygon.is_valid:
                                continue
                        elif new_polygon.geom_type is 'MultiPolygon':
                            polygons = []
                            for single_polygon in new_polygon.geoms:
                                if not single_polygon.is_valid or single_polygon.is_empty:
                                    continue
                                new_single_polygon = self.proc_polygon(single_polygon, ego_SE3_city)
                                if not new_single_polygon.is_valid:
                                    continue
                                polygons.append(new_single_polygon)
                            if len(polygons) == 0:
                                continue
                            new_polygon = MultiPolygon(polygons)
                            if not new_polygon.is_valid:
                                continue
                        else:
                            raise ValueError('{} is not valid'.format(new_polygon.geom_type))
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
            else:
                raise ValueError(f'WRONG type: line in boundary')
        map_boundary_geom.append(('boundary', polygon_list))
        return map_boundary_geom

    def get_map_ped_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        map_ped_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        # import pdb;pdb.set_trace()
        polygon_list = []
        for pc in avm:
            exterior_coords = pc
            interiors = []
            polygon = Polygon(exterior_coords, interiors)
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, polygon_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, polygon_list)

        map_ped_geom.append(('ped_crossing', polygon_list))
        return map_ped_geom

    def proc_line(self, line, ego_SE3_city):
        # import pdb;pdb.set_trace()
        new_line_pts_cityframe = np.array(list(line.coords))
        new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(new_line_pts_cityframe)
        line = LineString(new_line_pts_egoframe[:, :2])  # TODO
        return line


    
    def get_map_divider_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        map_divider_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for ls in avm:#avm:
            line = LineString(ls)
            if line.is_empty:  # Skip lines without nodes.
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                # import pdb;pdb.set_trace()
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line.geoms:
                        if single_line.is_empty:
                            continue

                        single_line = self.proc_line(single_line, ego_SE3_city)
                        line_list.append(single_line)
                else:
                    new_line = self.proc_line(new_line, ego_SE3_city)
                    line_list.append(new_line)
          
        map_divider_geom.append(('divider', line_list))
        return map_divider_geom 

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_instances.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_instances

   
    def bound_poly_geoms_to_instances(self, polygon_geom):
        # roads = polygon_geom[0][1]
        # lanes = polygon_geom[1][1]
        # union_roads = ops.unary_union(roads)
        # union_lanes = ops.unary_union(lanes)
        # union_segments = ops.unary_union([union_roads, union_lanes])
        # import pdb;pdb.set_trace()
        roads = polygon_geom[0][1]
        #lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        #union_lanes = ops.unary_union(lanes), union_lanes
        union_segments = ops.unary_union([union_roads])
        
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return  self._one_type_line_geom_to_instances(results)

    def line_geoms_to_instances(self, line_geom):
        lines = line_geom[0][1]
        multiline = MultiLineString(lines)
        union_lines = ops.unary_union(multiline)
        if union_lines.geom_type == 'LineString':
            return  self._one_type_line_geom_to_instances([union_lines])
        before_num = len(union_lines.geoms)
        # import pdb;pdb.set_trace()
        merged_lines = ops.linemerge(union_lines)
        if merged_lines.geom_type == 'LineString':
            return  self._one_type_line_geom_to_instances([merged_lines])
        after_num = len(merged_lines.geoms)
        # import pdb;pdb.set_trace()
        while after_num != before_num:
            before_num = len(merged_lines.geoms)
            merged_lines = ops.unary_union(merged_lines)
            if merged_lines.geom_type == 'LineString':
                break
            merged_lines = ops.linemerge(merged_lines)
            if merged_lines.geom_type == 'LineString':
                break
            after_num = len(merged_lines.geoms)
        
        return  self._one_type_line_geom_to_instances([merged_lines])

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid



class AV2MapExtractor(object):
    """Argoverse 2 map ground-truth extractor.

    Args:
        roi_size (tuple or list): bev range
        id2map (dict): log id to map json path
    """
    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 patch_size,
                 canvas_size,
                 id2map: Dict,
                 sample_dist=1,
                 num_samples=500,
                 padding=False,
                 normalize=False,
                 fixed_num=-1,
                 class2label={
                        'ped_crossing': 1,
                        'road_divider': 0,
                        'lane_divider': 0,
                        'divider': 0,
                        'boundary': 2,
                        'others': -1,
                    }) -> None:
        self.roi_size = roi_size
        self.id2map = {}
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

        self.class2label = class2label
        for log_id, path in id2map.items():
            self.id2map[log_id] = ArgoverseStaticMap.from_json(Path(path))
    
    def get_map_geom(self,
                     log_id, 
                     e2g_translation: NDArray, 
                     e2g_rotation: NDArray, 
                     polygon_ped=False) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `log_id` and ego pose.
        
        Args:
            log_id (str): log id
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global rotation matrix, shape (3, 3)
            polygon_ped: if True, organize each ped crossing as closed polylines. \
                Otherwise organize each ped crossing as two parallel polylines. \
                Default: True
        
        Returns:
            geometries (Dict): extracted geometries by category.
        '''
        avm = self.id2map[log_id]
        
        g2e_translation = e2g_rotation.T.dot(-e2g_translation)
        g2e_rotation = e2g_rotation.T
       
        
        roi_x, roi_y = self.roi_size[:2]
        local_patch = box(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)

        all_dividers = []
        # for every lane segment, extract its right/left boundaries as road dividers
        for _, ls in avm.vector_lane_segments.items():
            # right divider
            right_xyz = ls.right_lane_boundary.xyz
            right_mark_type = ls.right_mark_type
            right_ego_xyz = transform_from(right_xyz, g2e_translation, g2e_rotation)

            right_line = LineString(right_ego_xyz)
            right_line_local = right_line.intersection(local_patch)

            if not right_line_local.is_empty and not right_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(right_line_local)
                
            # left divider
            left_xyz = ls.left_lane_boundary.xyz
            left_mark_type = ls.left_mark_type
            left_ego_xyz = transform_from(left_xyz, g2e_translation, g2e_rotation)

            left_line = LineString(left_ego_xyz)
            left_line_local = left_line.intersection(local_patch)

            if not left_line_local.is_empty and not left_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(left_line_local)
        
        # remove repeated dividers since each divider in argoverse2 is mentioned twice
        # by both left lane and right lane
        all_dividers = remove_repeated_lines(all_dividers)
        
        ped_crossings = [] 
        for _, pc in avm.vector_pedestrian_crossings.items():
            edge1_xyz = pc.edge1.xyz
            edge2_xyz = pc.edge2.xyz
            ego1_xyz = transform_from(edge1_xyz, g2e_translation, g2e_rotation)
            ego2_xyz = transform_from(edge2_xyz, g2e_translation, g2e_rotation)

            # if True, organize each ped crossing as closed polylines. 
            if polygon_ped:
                vertices = np.concatenate([ego1_xyz, ego2_xyz[::-1, :]])
                p = Polygon(vertices)
                line = get_ped_crossing_contour(p, local_patch)
                if line is not None:
                    ped_crossings.append(line)

            # Otherwise organize each ped crossing as two parallel polylines.
            else:
                line1 = LineString(ego1_xyz)
                line2 = LineString(ego2_xyz)
                line1_local = line1.intersection(local_patch)
                line2_local = line2.intersection(local_patch)

                # take the whole ped cross if all two edges are in roi range
                if not line1_local.is_empty and not line2_local.is_empty:
                    ped_crossings.append(line1_local)
                    ped_crossings.append(line2_local)

        drivable_areas = []
        for _, da in avm.vector_drivable_areas.items():
            polygon_xyz = da.xyz
            ego_xyz = transform_from(polygon_xyz, g2e_translation, g2e_rotation)
            polygon = Polygon(ego_xyz)
            polygon_local = polygon.intersection(local_patch)

            drivable_areas.append(polygon_local)

        # union all drivable areas polygon
        drivable_areas = ops.unary_union(drivable_areas)
        drivable_areas = split_collections(drivable_areas)

        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # some dividers overlaps with boundaries in argoverse2 dataset
        # we need to remove these dividers
        all_dividers = remove_boundary_dividers(all_dividers, boundaries)

        # some dividers are split into multiple small parts
        # we connect these lines
        all_dividers = connect_lines(all_dividers)


        return dict(
            divider= self._one_type_line_geom_to_vectors( all_dividers) , # List[LineString]
            ped_crossing=self._one_type_line_geom_to_vectors( ped_crossings), # List[LineString]
            boundary=self._one_type_line_geom_to_vectors( boundaries) , # List[LineString]
            #drivable_area=drivable_areas, # List[Polygon],
        )

    def gen_vectorized_samples(self,
                     log_id: str, 
                     e2g_translation: NDArray, 
                     e2g_rotation: NDArray, 
                     polygon_ped=False) -> Dict[str, List[Union[LineString, Polygon]]]:
       
       
        results = self.get_map_geom(log_id,   e2g_translation,   e2g_rotation,   polygon_ped) 

        vectors = []
      
        for line, length in results['divider']:
            vectors.append((line.astype(float), length, self.class2label.get('divider', -1)))

        #print('divider')
        for ped_line, length in results['ped_crossing']:
            vectors.append((ped_line.astype(float), length, self.class2label.get('ped_crossing', -1)))
        #print('ped_crossing')
        for contour, length in results['boundary']:
            vectors.append((contour.astype(float), length, self.class2label.get('boundary', -1)))
        #print('contours')
        # filter out -1
        filtered_vectors = []
        for pts, pts_num, _type in vectors:
            if _type != -1:
                filtered_vectors.append({'pts': pts, 'pts_num': pts_num, 'type': _type})

        return filtered_vectors

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(l))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors
  

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances])
            #print(sampled_points.shape)
            sampled_points = sampled_points[:, :, :2].reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances])
            sampled_points = sampled_points[:, :, :2].reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid

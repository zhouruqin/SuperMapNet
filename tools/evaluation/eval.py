import os
import sys
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from tools.evaluation.ap import instance_mask_ap as get_batch_ap



def compute_one_ap( dt_masks, dt_scores, gt_masks, num_classes=3, map_resolution=(0.15, 0.15)):
    THRESHOLDS = [0.2, 0.5, 1.0, 1.5]
    SAMPLED_RECALLS = torch.linspace(0.1, 1, 10).cuda()
    map_resolution = map_resolution
    ap_matrix = torch.zeros((num_classes, len(THRESHOLDS))).cuda()
    ap_count_matrix = torch.zeros((num_classes, len(THRESHOLDS))).cuda()
    
    print(torch.from_numpy(dt_masks).cuda().size(),  gt_masks.cuda().size(), torch.from_numpy(np.array(dt_scores)).cuda().size())
    ap_matrix, ap_count_matrix = get_batch_ap(
                ap_matrix,
                ap_count_matrix,
                torch.from_numpy(dt_masks).cuda(),
                gt_masks.cuda(),
                *map_resolution,
                torch.from_numpy(np.array(dt_scores)).cuda(),
                THRESHOLDS,
                SAMPLED_RECALLS,
            )
    return ap_matrix, ap_count_matrix

def format_print( ap_matrix):
    CLASS_NAMES = ["Divider", "PedCross", "Contour"]
    res_matrix = []
    table_header = ["Class", "AP@.2", "AP@.5", "AP@1.", "AP@1.5", "mAP@HARD", "mAP@EASY"]
    table_values = []
    for i, cls_name in enumerate(CLASS_NAMES):
        res_matrix_line = [ap_matrix[i][0], ap_matrix[i][1], ap_matrix[i][2], ap_matrix[i][3], np.mean(ap_matrix[i][:-1]), np.mean(ap_matrix[i][1:])]
        res_matrix.append(res_matrix_line)
        table_values.append([cls_name] +  line_data_to_str(*res_matrix_line))
    avg = np.mean(np.array(res_matrix), axis=0)
    table_values.append(["Average", *line_data_to_str(*avg)])
    table_str = tabulate(table_values, headers=table_header, tablefmt="grid")
    print(table_str)
    return table_str


def line_data_to_str(ap0, ap1, ap2, ap3, map1, map2):
    return [
        "{:.1f}".format(ap0 * 100),
        "{:.1f}".format(ap1 * 100),
        "{:.1f}".format(ap2 * 100),
        "{:.1f}".format(ap3 * 100),
        "{:.1f}".format(map1 * 100),
        "{:.1f}".format(map2 * 100),
    ]



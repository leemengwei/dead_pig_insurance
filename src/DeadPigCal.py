import json
import pandas as pd
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class Line:
    def __init__(self, p1, p2):
        self.x1 = p1.x
        self.x2 = p2.x
        self.y1 = p1.y
        self.y2 = p2.y


def cross_point(l1, l2):  # 计算交点函数
    # l1
    x1 = l1.x1
    y1 = l1.y1
    x2 = l1.x2
    y2 = l1.y2
    # l2
    x3 = l2.x1
    y3 = l2.y1
    x4 = l2.x2
    y4 = l2.y2
    # l1斜率不存在
    if abs(x1 - x2) < 1e-7:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - x1 * k1
    # l2斜率不存在
    if abs(x3 - x4) < 1e-7:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) / (x4 - x3)  # 斜率存在操作
        b2 = y3 - x3 * k2
    if k1 is None and k2 is None:
        raise TypeError("两直线平行无交点")
    elif k1 is None:
        x = x1
        y = k2 * x + b2
    elif k2 is None:
        x = x3
        y = k1 * x + b1
    elif abs(k1 - k2) < 1e-7:
        raise TypeError("两直线平行无交点")
    else:
        x = (b1 - b2) / (k2 - k1)
        y = k1 * x + b1
    return Point(x, y)


def dis(line):
    x1 = line.x1
    y1 = line.y1
    x2 = line.x2
    y2 = line.y2
    return (pow(x1 - x2, 2) + pow(y1 - y2, 2)) ** 0.5


# 交点 / 原线段
# 输入 快递left两点坐标（顺序无关） 快递right两点坐标（顺序无关） 猪上两点坐标（顺序无关）
def cal(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    p3 = Point(x3, y3)
    p4 = Point(x4, y4)
    p5 = Point(x5, y5)
    p6 = Point(x6, y6)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    l3 = Line(p5, p6)
    p7 = cross_point(l1, l3)
    p8 = cross_point(l2, l3)
    l4 = Line(p7, p8)
    return dis(l4) / dis(l3)


if __name__ == '__main__':
    # print(cal(1142, 1376, 1118, 2350, 2506, 1388, 2506, 2380, 1004, 763, 2374, 901))
    df = pd.read_csv('via_project_dead_pig_csv.csv')
    new_df = pd.DataFrame(
        columns=['file_name_top', 'file_name_side',
                 'nose_coordinate_top', 'ear_coordinate_top', 'hip_coordinate_top', 'tail_coordinate_top',
                 'left_top_coordinate_top', 'left_bottom_coordinate_top', 'right_top_coordinate_top', 'right_bottom_coordinate_top',
                 'nose_coordinate_side', 'ear_coordinate_side', 'hip_coordinate_side', 'tail_coordinate_side',
                 'left_top_coordinate_side', 'left_bottom_coordinate_side', 'right_top_coordinate_side', 'right_bottom_coordinate_side',
                 'nose_hip_top', 'nose_tail_top', 'ear_hip_top', 'ear_tail_top',
                 'nose_hip_side', 'nose_tail_side', 'ear_hip_side', 'ear_tail_side',
                 'length', 'weight'
                 ])
    flag = {}
    index_cnt = 0
    for index, row in df.iterrows():
        info = row['filename'].split('_')
        if info[1] == 'side':
            continue
        # 开始计数
        if row['region_id'] == 0:
            new_df = new_df.append({'file_name_' + info[1]: row['filename'], 'length': int(info[0]), 'weight': int(info[2])}, ignore_index=True)
            flag[info[0] + info[2] + info[3] + info[4]] = index_cnt
            index_cnt += 1
        new_df[row['region_attributes'][16:-8] + '_coordinate_' + info[1]][index_cnt - 1] = [int(json.loads(row['region_shape_attributes'])['cx']), int(json.loads(row['region_shape_attributes'])['cy'])]
    for index, row in df.iterrows():
        info = row['filename'].split('_')
        if info[1] == 'top':
            continue
        # 开始计数
        if row['region_id'] == 0:
            index_cnt = flag[info[0] + info[2] + info[3] + info[4]]
            new_df['file_name_' + info[1]][index_cnt] = row['filename']
        new_df[row['region_attributes'][16:-8] + '_coordinate_' + info[1]][index_cnt] = [int(json.loads(row['region_shape_attributes'])['cx']), int(json.loads(row['region_shape_attributes'])['cy'])]
    lst = ['nose_hip_top', 'nose_tail_top', 'ear_hip_top', 'ear_tail_top',
           'nose_hip_side', 'nose_tail_side', 'ear_hip_side', 'ear_tail_side']
    for index, row in new_df.iterrows():
        for i in lst:
            info = i.split('_')
            try:
                if (not pd.isna(row[info[0] + '_coordinate_' + info[2]])[0]) and (not pd.isna(row[info[1] + '_coordinate_' + info[2]])[0]):
                    new_df[i][index] = cal(row['left_top_coordinate_' + info[2]][0], row['left_top_coordinate_' + info[2]][1],
                                           row['left_bottom_coordinate_' + info[2]][0], row['left_bottom_coordinate_' + info[2]][1],
                                           row['right_top_coordinate_' + info[2]][0], row['right_top_coordinate_' + info[2]][1],
                                           row['right_bottom_coordinate_' + info[2]][0], row['right_bottom_coordinate_' + info[2]][1],
                                           row[info[0] + '_coordinate_' + info[2]][0], row[info[0] + '_coordinate_' + info[2]][1],
                                           row[info[1] + '_coordinate_' + info[2]][0], row[info[1] + '_coordinate_' + info[2]][1])
            except TypeError:
                pass
    for index, row in new_df.iterrows():
        if new_df['weight'][index] != 0:
            new_df['weight'][index] = float(new_df['weight'][index]) / 100
    new_df.to_csv('dead_pig.csv')

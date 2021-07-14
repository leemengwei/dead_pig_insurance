import os
import PIL
import math
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


if __name__ == '__main__':
    # 图片文件夹目录
    path = "image"
    files = os.listdir(path)
    for img_file in files:
        print(img_file)
        # 读入图片
        img = Image.open(path + '/' + img_file)
        # 指定画笔
        draw = ImageDraw.Draw(img)
        xx = np.array(img).shape[1]
        yy = np.array(img).shape[0]
        # 加载字体
        font = ImageFont.truetype('STSONG.ttf', int(xx / 36))
        (x1, y1) = (int(xx / 36), int(yy / 27))
        (x2, y2) = (int(xx / 36), 2 * int(yy / 27))
        (x3, y3) = (int(xx / 36), 3 * int(yy / 27))
        (x4, y4) = (int(xx / 36), 4 * int(yy / 27))
        rgb = (255, 255, 255)
        df = pd.read_csv('dead_pig_predict.csv')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for i in range(len(df)):
            if df['file_name_top'].iloc[i] == img_file or df['file_name_side'].iloc[i] == img_file:
                img_index = i
                break
        point_lst = ['nose_coordinate_', 'ear_coordinate_', 'hip_coordinate_', 'tail_coordinate_',
                     'left_top_coordinate_', 'left_bottom_coordinate_', 'right_top_coordinate_', 'right_bottom_coordinate_']
        point_cnt = 8
        info = img_file.split('_')
        for k in point_lst:
            col = k + info[1]
            if pd.isna(df[col].iloc[img_index]):
                point_cnt -= 1
                continue
            center = (df[col].iloc[img_index])[1:-1].split(', ')
            for i in range(len(center)):
                center[i] = int(center[i])
            r = int(xx / 180)
            for i in range(center[0] - r, center[0] + r + 1):
                for j in range(center[1] - r, center[1] + r + 1):
                    if (pow(i - center[0], 2) + pow(j - center[1], 2)) ** 0.5 <= r:
                        draw.point((i, j), fill=rgb)
        is_output = 1
        if pd.isna(df['predict_length'].iloc[img_index]) or pd.isna(df['predict_weight'].iloc[img_index]):
            is_output = 0
        if is_output:
            text1 = "识别结果"
            draw.text((x1, y1), text1, fill=rgb, font=font)
            text2 = "状态点 : %d" % point_cnt
            draw.text((x2, y2), text2, fill=rgb, font=font)
            length = float(df['predict_length'].iloc[img_index])
            text3 = "体长 : %.2fCM" % length
            draw.text((x3, y3), text3, fill=rgb, font=font)
            weight = float(df['predict_weight'].iloc[img_index])
            text4 = "体重 : %.2fKG" % weight
            draw.text((x4, y4), text4, fill=rgb, font=font)

            img.save('image_output/' + img_file)
        del draw  # 删除画笔
        img.close()  # 关闭图片

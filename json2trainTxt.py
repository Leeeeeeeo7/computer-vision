import json
import os
import shutil
import cv2
import os
from numpy.lib.twodim_base import triu_indices_from
import pandas as pd
from glob import glob
import codecs

print(cv2.__version__)


def getBoundingBox(points):              # 找到两个点的x最大最小值，y最大最小值
    xmin = points[0][0]
    xmax = points[0][0]
    ymin = points[0][1]
    ymax = points[0][1]
    for p in points:
        if p[0] > xmax:
            xmax = p[0]
        elif p[0] < xmin:
            xmin = p[0]

        if p[1] > ymax:
            ymax = p[1]
        elif p[1] < ymin:
            ymin = p[1]
    return [int(xmin), int(xmax), int(ymin), int(ymax)]


def json2txt(json_path, midTxt_path):
    json_data = json.load(open(json_path))         # 读取并加载json文件
    img_h = json_data["imageHeight"]
    img_w = json_data["imageWidth"]
    shape_data = json_data["shapes"]
    shape_data_len = len(shape_data)
    img_name = os.path.split(json_path)[-1].split(".json")[0]    # 以“PATH”中最后一个‘/’作为分隔符
    name = img_name + '.jpg'                            # 找到json文件对应的图片
    data = ''
    for i in range(shape_data_len):
        lable_name = shape_data[i]["label"]             # 获取一个json文件里的每个标签的名称
        points = shape_data[i]["points"]                # 获取一个json文件里的每个标签的坐标位置
        [xmin, xmax, ymin, ymax] = getBoundingBox(points)
        if xmin <= 0:
            xmin = 0
        if ymin <= 0:
            ymin = 0
        if xmax >= img_w:  
            xmax = img_w - 1
        if ymax >= img_h:
            ymax = img_h - 1
        b = name + ' ' + lable_name + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
        print(b)
        data += b + '\n'
    with open(midTxt_path + '/' + img_name + ".txt", 'w', encoding='utf-8') as f:    
        f.writelines(data)         # 创建文件并写入相关数据 图片文件名，每个图片的标签名、xy最大最小指
                                                   
                    

def txt2darknet(midTxt_path, img_path):
    data = pd.DataFrame()
    filelist = os.listdir(midTxt_path) 
    for file in filelist:                                                   
        file_path = os.path.join(midTxt_path, file)
        filename = os.path.splitext(file)[0]
        imgName = filename + '.jpg'
        imgPath = os.path.join(img_path, imgName)
        # for path in img_path:
        #     imgPath = os.path.join(path, imgName)
        #     if not os.path.exists(imgPath):
        #         continue
        #     else:
        #         break
        
        if not os.path.exists(imgPath):
            imgName = filename + '.png'
            imgPath = os.path.join(img_path, imgName)
            if not os.path.exists(imgPath):
                imgName = filename + '.jpeg'
                imgPath = os.path.join(img_path, imgName)
        img = cv2.imread(imgPath)
        print(imgPath)
        [img_h, img_w, _] = img.shape
        data = ""
        with codecs.open(file_path, 'r', encoding='utf-8',errors='ignore') as f1:
            for line in f1.readlines():
                line = line.strip('\n')
                a = line.split(' ')
                if int(a[5]) - int(a[3]) <= 15 or int(a[4]) - int(a[2]) <= 15:
                    img[int(a[3]):int(a[5]), int(a[2]):int(a[4]), :] = (0,0,0)
                    continue
                if a[1] == 'other' or a[1] == 'mask' or a[1] == 'del':
                    img[int(a[3]):int(a[5]), int(a[2]):int(a[4]), :] = (0,0,0)
                    continue
                if a[1] == 'head': 
                    a[1] = '0'
                elif a[1] == 'hat':
                    a[1] = '1'
                elif a[1] == 'helmet':
                    a[1] = '2'
                elif a[1] == 'locap':
                    a[1] = '3'
                elif a[1] == 'blueworkclothes':
                    a[1] = '4'
                elif a[1] == 'cloth' or a[1] == 'clothes':
                    a[1] = '5'
                elif a[1] == 'halfcloth':
                    a[1] = '6'
                elif a[1] == 'saferope':
                    a[1] = '7'
                elif a[1] == 'refvest':
                    a[1] = '8'
                elif a[1] == 'apron':
                    a[1] = '9'
                elif a[1] == 'wearmask':
                    a[1] = '10'
                elif a[1] == 'whiteworkclothes':
                    a[1] = '11'
                elif a[1] == 'whiteapron':
                    a[1] = '12'
                elif a[1] == 'chefhat':
                    a[1] = '13'
                elif a[1] == 'workhat':
                    a[1] = '14'


                x1 = float(a[2])
                y1 = float(a[3])
                w = float(a[4]) - float(a[2])
                h = float(a[5]) - float(a[3])

                # if w <= 15 and h <= 15: continue

                center_x = float(a[2]) + w / 2
                center_y = float(a[3]) + h / 2
                a[2] = str(center_x / img_w)
                a[3] = str(center_y / img_h)
                a[4] = str(w / img_w)
                a[5] = str(h / img_h)
                b = a[1] + ' ' + a[2] + ' ' + a[3] + ' ' + a[4] + ' ' + a[5]
                print(b)
                data += b + '\n'
        with open(saved_path + '/' + filename + ".txt", 'w', encoding='utf-8') as f2:    
            f2.writelines(data)

json_path = "存放json的文件夹"
midTxt_path = "随便创建一个temp文件夹，会自动删掉"
img_path = "存放图片的文件夹"
saved_path = '保存训练格式的文件的文件夹'

if not os.path.exists(midTxt_path):
    os.mkdir(midTxt_path)

filelist = os.listdir(json_path)                    # 取文件夹下的所有文件或文件夹
for file in filelist:
    old_dir = os.path.join(json_path, file)         # 路径与每个文件（夹）拼接
    if os.path.isdir(old_dir):
        continue                                    # 如果是文件夹跳过，只要文件
    filetype = os.path.splitext(file)[1]            
    if(filetype != ".json"): continue               #取文件后缀，只要json格式的
    json2txt(old_dir, midTxt_path)

txt2darknet(midTxt_path, img_path)
shutil.rmtree(midTxt_path)

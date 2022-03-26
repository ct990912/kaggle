# written by Cai
import os
import json
import random
import shutil

def getFile(root_path):
    file_list = list()
    for file in os.listdir(root_path):
        if ('.' in file) == False and file != "coco":
            file_list.append(file)
    return file_list


def getAllJson(path):
    file_list = list()
    for file in os.listdir(path):
        if file.split('.')[1] != "vott":
            file_list.append(path + '/' + file)
    return file_list


def read_json(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def getImg(path):
    img_list = list()
    for img_file in os.listdir(path):
        if img_file != "annotations":
            img_list.append(img_file)
    return img_list
 
 
def convert2coco(root_path, split_rate=0.1,saveFile=None):
    """
    Args:
        root_path(str): your data's root path, e.g., "/home/cai/data/ArTaxOr/".
        split_rate(float): Divide the dataset into training and validation sets.
        saveFile(str): if saveFile is None, converted data will store \
                in 'root_path/coco/'; if given, converted data will \
                store your own "saveFile".
    Examples:
        >>>convert2coco("/home/cai/data/ArTaxOr/", 0.1, "/home/cai/coco/")
    """
    if saveFile is None:
        saveFile=root_path+"coco/"
        if not os.path.exists(root_path + "coco/"):
            os.makedirs(root_path + "coco/")
        if not os.path.exists(root_path + "coco/annotations/"):
            os.makedirs(root_path + "coco/annotations/")
        if not os.path.exists(root_path + "coco/train/"):
            os.makedirs(root_path + "coco/train/")
        if not os.path.exists(root_path + "coco/val/"):
            os.makedirs(root_path + "coco/val/")
    else:
        if not os.path.exists(saveFile + "annotations/"):
            os.makedirs(saveFile + "annotations/")
        if not os.path.exists(saveFile + "train/"):
            os.makedirs(saveFile + "train/")
        if not os.path.exists(saveFile + "val/"):
            os.makedirs(saveFile + "val/")
        
    all_class_file = getFile(root_path)  # 分别获取每个分类的目录名，存在list里面
    coco_train_annotations = dict(images=list(), annotations=list(), categories=list())  # coco训练集annotations
    coco_val_annotations = dict(images=list(), annotations=list(), categories=list())  # coco验证集annotations
    coco_annotations = [coco_train_annotations, coco_val_annotations]  # 用于划分测试集和训练集
    img_idx = 10
    bbox_idx = 100000
    label_idx = 0
    # ----------------获取coco格式中的categories
    for idx, label in enumerate(all_class_file):
        category = dict(id=idx, supercategory="Arthropod", name=label)
        coco_annotations[0]["categories"].append(category)
        coco_annotations[1]["categories"].append(category)

    # ---------------获取coco格式中的images和annotations
    for each_class in all_class_file:
        print("start to convert "+each_class+'\n')
        root_img = root_path + each_class + '/'  # 某类图片的根路径
        all_annotations = getAllJson(root_path + each_class + "/annotations")  # 获取所有的json annotation
        for json_file in all_annotations:  # 访问单个json，并按splite_rate数据存入到coco_anotations中
            data = read_json(json_file)
            flag = 1 if random.random() < split_rate else 0  # 划分测试集和训练集 train:0; val: 1;
            shutil.copyfile(root_img + data["asset"]["name"],
                        saveFile + ("train/" if flag == 0 else "val/") + data["asset"]["name"])
            # ---------------获取coco格式中的image
            img = dict(file_name=data["asset"]["name"],
                       height=data["asset"]["size"]["height"],
                       width=data["asset"]["size"]["width"],
                       id=img_idx)
            coco_annotations[flag]["images"].append(img)
            #  [x, y, w, h]， 下面的代码是一张图里面的所有bbox append到annotations里面
            for region in data["regions"]:
                bbox = region["boundingBox"]
                anno = dict(image_id=img_idx, segmentation=[[]], area=240, iscrowd=0,
                            bbox=[bbox["left"], bbox["top"], bbox["width"], bbox["height"]],
                            id=bbox_idx, category_id=label_idx)
                bbox_idx += 1
                coco_annotations[flag]["annotations"].append(anno)

            img_idx += 1
        label_idx += 1

    with open(saveFile+"annotations/train.json", "w") as f:
        json.dump(coco_annotations[0], f)
    with open(saveFile + "annotations/val.json", "w") as f:
        json.dump(coco_annotations[1], f)
    print("finish...")
    
if __name__ == "__main__":
    convert2coco("../input/arthropod-taxonomy-orders-object-detection-dataset/ArTaxOr/",0.1,"./")

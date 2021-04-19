import os
import random
import shutil
from imutils import paths
import glob
from tqdm import tqdm
import json
import time


def change_label(json_file):
    with open(json_file, 'r', encoding='gbk') as rf:
        data = json.load(rf)
        data['date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        for i, info in enumerate(data["info"]):
            for key in data['info'][i]['pts'].keys():
                # 设置 z 值, 用来表示该点的置信度, 后续评测指标时会用到
                data['info'][i]['pts'][key]['z'] = 1.0
    return data

def split_train_val_from_paths(root, out_root, ratio=0.1):
    # 创建目录
    out_train = os.path.join(out_root, "train")
    out_train_images = os.path.join(out_train, "images")
    out_train_labels = os.path.join(out_train, "labels")

    out_val = os.path.join(out_root, "val")
    out_val_images = os.path.join(out_val, "images")
    out_val_labels = os.path.join(out_val, "labels")

    if not os.path.exists(out_train):
        os.makedirs(out_train)
    if not os.path.exists(out_train_images):
        os.makedirs(out_train_images)
    if not os.path.exists(out_train_labels):
        os.makedirs(out_train_labels)

    if not os.path.exists(out_val):
        os.makedirs(out_val)
    if not os.path.exists(out_val_images):
        os.makedirs(out_val_images)
    if not os.path.exists(out_val_labels):
        os.makedirs(out_val_labels)

    # 随机化数据
    im_paths = list(paths.list_images(os.path.join(root, "images")))
    random.shuffle(im_paths)
    num = len(im_paths)
    test_len = int(ratio * num)

    im_train, im_test = im_paths[test_len:], im_paths[:test_len]

    # 拷贝数据
    for i, imp in tqdm(enumerate(im_train), total=num - test_len):
        lbp = imp.replace("images", "labels").replace("jpg", "json")
        shutil.copy2(imp, os.path.join(out_train_images, os.path.basename(imp)))
        # 处理 json 数据, 并保存到目标位置
        result = change_label(lbp)
        with open(os.path.join(out_train_labels, os.path.basename(lbp)),'w',encoding='utf-8') as wf:
            json.dump(result, wf, ensure_ascii=False, indent=4)
        # shutil.copy2(lbp, os.path.join(out_train_labels, os.path.basename(lbp)))

    for i, imp in tqdm(enumerate(im_test), total=test_len):
        lbp = imp.replace("images", "labels").replace("jpg", "json")
        shutil.copy2(imp, os.path.join(out_val_images, os.path.basename(imp)))
        # 处理 json 数据, 并保存到目标位置
        result = change_label(lbp)
        with open(os.path.join(out_val_labels, os.path.basename(lbp)),'w',encoding='utf-8') as wf:
            json.dump(result, wf, ensure_ascii=False, indent=4)
        # shutil.copy2(lbp, os.path.join(out_val_labels, os.path.basename(lbp)))


def main():
    root = "/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1"
    out_root = "/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1"

    split_train_val_from_paths(root, out_root, ratio=0.05)


if __name__ == '__main__':
    main()
    

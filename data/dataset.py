import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
import random
import config as cfg
import os.path as osp
from imutils import paths
import cv2
import numpy as np
from PIL import Image
import json


def read_image(img_path, part_size=0, rand_ch=None):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if img_path == 'no':
        h = part_size[1]
        w = part_size[0]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')
    else:
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')

                if rand_ch and random.random() > rand_ch:
                    img = Image.fromarray(np.asarray(img)[:, :, ::-1])

                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path))
                pass
        return img


class KeypointsDetDataSet(Dataset):
    def __init__(self, root, input_size=(256, 256), is_train=False, transform=None, data_aug=None):
        super(KeypointsDetDataSet, self).__init__()

        """ 数据需处理成一图一目标
        """
        # get image paths list
        self.image_paths = list(paths.list_images(root))
        random.shuffle(self.image_paths)
        self.is_train = is_train
        self.input_size = input_size
        self.data_aug = data_aug
        if transform is None:
            self.transform = T.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_p = self.image_paths[index]
        lb_p = img_p.replace("images", "labels").replace("jpg", "json")

        # read label
        with open(lb_p, 'r', encoding='utf-8') as rf:
            info_pts = json.load(rf)['info'][0]
        # points[:, 2]用于处理遮挡关键点问题, 加一个类别loss, 后续的数据增强可能会改变某些点的类别
        points = np.array([[p_v for p_k, p_v in v.items()]
                           for k, v in info_pts['pts'].items()])[:, :2]

        # read image
        img = cv2.imread(img_p)

        # 切出手部截图
        x_, y_, w_, h_ = cv2.boundingRect(points.astype(np.int32))

        # ==== Augments ====
        if self.is_train:
            # 手部区域截图抖动
            side = max(w_, h_) * (1. + float(random.randint(5, 50)) / 100.)
        else:
            # 验证时基于关键点提供的框固定放大手部 1.1 倍
            side = max(w_, h_) * 1.1
        x_mid = x_ + (w_ // 2)
        y_mid = y_ + (h_ // 2)

        x1, y1, x2, y2 = int(x_mid-(side/2.)), int(y_mid-(side/2.)), int(x_mid+(side/2.)), int(y_mid+(side/2.))

        x1 = np.clip(x1, 0, img.shape[1]-1)
        x2 = np.clip(x2, 0, img.shape[1]-1)

        y1 = np.clip(y1, 0, img.shape[0]-1)
        y2 = np.clip(y2, 0, img.shape[0]-1)

        # 截取对象区域
        img = img[y1: y2, x1: x2, :]

        # 默认关键点的位置基于原始图像, 后续再优化
        points[:, 0] -= x1
        points[:, 1] -= y1
        if self.data_aug:
            img, points = self.data_aug(images=np.expand_dims(img, axis=0), keypoints=np.expand_dims(points, axis=0))
            img = img.squeeze(0)
            points = points.squeeze(0)
        h, w, _ = img.shape
        # normalized points 0 - 1
        points[:, 0] /= w
        points[:, 1] /= h

        # # 可视化检查数据
        # from tools.utils import visualisation
        # vis_points = points.copy()
        # vis_points[:, 0] *= w
        # vis_points[:, 1] *= h
        # obj_img = img.copy().astype(np.uint8)
        # # for p in vis_points.astype(np.int32):
        # #     cv2.circle(obj_img, tuple(p), 1, (0, 0, 255), 2)
        # obj_img = visualisation(obj_img, vis_points.astype(np.int32))
        # cv2.imwrite("out.jpg", obj_img)
        # exit()

        img = Image.fromarray(img[:, :, ::-1])  # bgr -> rgb for PIL
        data = self.transform(img)
        # img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        # data = (img-128.)/256.
        # data = torch.FloatTensor(data.transpose([2, 0, 1]))

        return data, torch.from_numpy(points.reshape(-1)).float()

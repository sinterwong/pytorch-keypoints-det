import torch
from utils import progress_bar
from models.resnet import resnet18
from models.repvgg import get_RepVGG_func_by_name
import config as cfg
from data.transform import data_transform
from data.dataset import ImageDataSet
import cv2
import numpy as np
import shutil
import os


def test(model_path, val_path, device='cuda', out_err="data/error"):
    # class map
    class_dict = {v: k for k, v in dict(enumerate(cfg.classes)).items()}
    idx2classes = dict(enumerate(cfg.classes))

    # create dataloader
    transform_test = data_transform(False)
    testset = ImageDataSet(root=cfg.val_root, classes_dict=class_dict, transform=transform_test, is_train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # loading model
    # model_info = torch.load(model_path)
    # net = resnet18(pretrained=False, num_classes=len(cfg.classes))
    # net.load_state_dict(model_info["net"])
    repvgg_build_func = get_RepVGG_func_by_name("RepVGG-A0")
    net = repvgg_build_func(num_classes=len(cfg.classes), pretrained_path=None, deploy=False)
    
    model_info = torch.load(model_path)
    net.load_state_dict(model_info["net"])
    net = net.to(device)
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, im_paths) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            result = predicted.eq(targets)
            correct += result.sum().item()

            # 保存预测错误的图片
            if out_err:
                if not os.path.exists(out_err):
                    os.makedirs(out_err)
                im_paths = np.array(im_paths)
                err_indexes = ~np.array(result.to('cpu'))
                err_im_paths = im_paths[err_indexes]
                right_result = [idx2classes[i] for i in np.array(targets.to('cpu'))[err_indexes]]
                err_result = [idx2classes[i] for i in np.array(predicted.to('cpu'))[err_indexes]]

                for (p, r, e) in zip(err_im_paths, right_result, err_result):
                    out_dir = os.path.join(out_err, e, r)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))


if __name__ == "__main__":
    model_path = 'checkpoint/best.pth'
    val_path = '/home/wangjq/wangxt/datasets/gesture-dataset/hand_gesture_v1/val'
    test(model_path, val_path)

import torch
from tools.utils import progress_bar
from models.get_network import build_network_by_name
import config as cfg
from data.transform import fast_transform
from data.dataset import ImageDataSet, KeypointsDetDataSet
import cv2
import numpy as np
import shutil
import os


def test(model_name, model_path, val_path, device='cuda'):

    # create dataloader
    transform = fast_transform()
    testset = KeypointsDetDataSet(root=cfg.val_root, input_size=cfg.input_size, is_train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # loading model
    net = build_network_by_name(model_name, None, num_classes=cfg.num_classes, deploy=True)
    
    model_info = torch.load(model_path)
    # net.load_state_dict(model_info['net'])
    net.load_state_dict(model_info)
    net = net.to(device)
    net.eval()

    total = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                total += targets.size(0)
                acc += 1 - torch.mean(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0))
            progress_bar(batch_idx, len(testloader), 'Acc(R square): %.3f%% (%d)' % (100.*acc/(batch_idx+1), total))


if __name__ == "__main__":
    model_name = 'resnet50'
    model_path = 'weights/resnet_50-size-256-loss-0.0642.pth'
    val_path = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/val'
    test(model_name, model_path, val_path)

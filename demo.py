import torch
from models.resnet import resnet10, resnet18, resnet34, resnet50
from models.seresnet import se_resnet10, se_resnet18, se_resnet34, se_resnet50
from models.repvgg import get_RepVGG_func_by_name
from models.mobilenetv3 import mobilenet_v3_small
from tools.utils import visualisation
import config as cfg
import cv2
import numpy as np
import shutil
import os


class Inference():
    def __init__(self, model, ckpt, input_size, device='cuda'):
        self.input_size = input_size
        self.device = device
        self.net = None
        if model == "resnet10":
            self.net = resnet10(pretrained=None, num_classes=cfg.num_classes)
        elif model == "resnet18":
            self.net = resnet18(pretrained=None, num_classes=cfg.num_classes)
        elif model == "resnet34":
            self.net = resnet34(pretrained=None, num_classes=cfg.num_classes)
        elif model == "resnet50":
            self.net = resnet50(pretrained=None, num_classes=cfg.num_classes)
        elif model == "seresnet10":
            self.net = se_resnet10(pretrained=None, num_classes=cfg.num_classes)
        elif model == "seresnet18":
            self.net = se_resnet18(pretrained=None, num_classes=cfg.num_classes)
        elif model == "seresnet34":
            self.net = se_resnet34(pretrained=None, num_classes=cfg.num_classes)
        elif model == "seresnet50":
            self.net = se_resnet50(pretrained=None, num_classes=cfg.num_classes)
        elif model == "mobilenetv3_small":
            self.net = mobilenet_v3_small(pretrained=None, num_classes=cfg.num_classes)
        elif model.split("-")[0] == "RepVGG":
            repvgg_build_func = get_RepVGG_func_by_name(model)
            self.net = repvgg_build_func(num_classes=cfg.num_classes, pretrained_path=None, deploy=True)
        else:
            raise Exception("暂未支持, 请在此处手动添加")
        self._load_model(ckpt)
    
    def _load_model(self, ckpt):
        model_info = torch.load(ckpt)
        self.net.load_state_dict(model_info["net"])
        self.net = self.net.to(self.device)
        self.net.eval()

    def single_image(self, im_p):
        img = cv2.imread(im_p)[:, :, ::-1]
        img = cv2.resize(img, (self.input_size[1], self.input_size[0])).astype(np.float32)
        show_img = img.copy()
        img /= 255.0
        data = torch.from_numpy(np.expand_dims(img.transpose([2, 0, 1]), axis=0)).to(self.device)
        with torch.no_grad():
            output = self.net(data).to('cpu').numpy()

        output = output.squeeze(0).reshape(-1, 2)
        output[:, 0] *= self.input_size[0]
        output[:, 1] *= self.input_size[1]
        
        # 可视化检查数据
        vis_points = output.astype(np.int32)
        show_img = visualisation(show_img, vis_points)
        cv2.imwrite("out.jpg", show_img[:, :, ::-1])

        # TODO

if __name__ == "__main__":
    model = 'resnet50'
    model_path = 'checkpoint/resnet50_handpose_224x224_88.pth'
    img_path = 'data/demo.jpg'
    engine = Inference(model, model_path, [224, 224])
    engine.single_image(img_path)

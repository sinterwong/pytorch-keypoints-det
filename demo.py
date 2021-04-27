import torch
from models.get_network import build_network_by_name
from tools.utils import visualisation
import config as cfg
import cv2
import numpy as np
import shutil
import os
from imutils import paths


class Inference():
    def __init__(self, model, ckpt, input_size, device='cuda'):
        self.input_size = input_size
        self.device = device
        self.net = build_network_by_name(model, None, num_classes=cfg.num_classes, deploy=True)
        self._load_model(ckpt)

    def _load_model(self, ckpt):
        model_info = torch.load(ckpt)
        self.net.load_state_dict(model_info["net"])
        self.net = self.net.to(self.device)
        self.net.eval()

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (self.input_size[1], self.input_size[0])).astype(np.float32)
        frame /= 255.0
        data = torch.from_numpy(np.expand_dims(frame.transpose([2, 0, 1]), axis=0)).to(self.device)
        return data

    def _single_image(self, frame):
        h, w, _ = frame.shape
        data = self._preprocess(frame)
        with torch.no_grad():
            output = self.net(data).to('cpu').numpy()
        output = output.squeeze(0).reshape(-1, 2)
        output[:, 0] *= w
        output[:, 1] *= h

        return output

    def vis_image_from_path(self, im_p):
        img = cv2.imread(im_p)
        show_img = img.copy()
        output = self._single_image(img[:, :, ::-1])

        vis_points = output.astype(np.int32)
        return visualisation(show_img, vis_points)

    def vis_images_from_dir(self, im_root, out_root):
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        image_paths = list(paths.list_images(im_root))[0:50:10]
        for i, p in enumerate(image_paths):
            show_img = self.vis_image_from_path(p)
            out_path = os.path.join(out_root, os.path.basename(p))
            cv2.imwrite(out_path, show_img)


def main():
    model = 'seresnet34'
    model_path = 'checkpoint/handpose/seresnet34/baseline/seresnet34_handpose_224x224_86.915.pth'
    engine = Inference(model, model_path, [224, 224])

    im_root = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/val'
    out_root = "data/%s_vis" % model

    engine.vis_images_from_dir(im_root, out_root)


if __name__ == "__main__":
    main()

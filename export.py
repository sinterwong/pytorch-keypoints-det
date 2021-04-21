from __future__ import print_function
from __future__ import division

import os
import torch
import collections
from models.resnet import resnet10, resnet18, resnet34, resnet50
from models.seresnet import se_resnet10, se_resnet18, se_resnet34, se_resnet50
from models.mobilenetv3 import mobilenet_v3_small
from models.repvgg import get_RepVGG_func_by_name, repvgg_model_convert
import config as cfg


def main():
    if cfg.model == "resnet10":
        model = resnet10(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "resnet18":
        model = resnet18(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "resnet34":
        model = resnet34(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "resnet50":
        model = resnet50(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet10":
        model = se_resnet10(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet18":
        model = se_resnet18(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet34":
        model = se_resnet34(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet50":
        model = se_resnet50(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "mobilenetv3_small":
        model = mobilenet_v3_small(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model.split("-")[0] == "RepVGG":
        repvgg_build_func = get_RepVGG_func_by_name(cfg.model)
        model = repvgg_build_func(num_classes=cfg.num_classes, pretrained_path=cfg.pretrained, deploy=False)

    else:
        raise Exception("暂未支持, 请在此处手动添加")

    model_path = "checkpoint/best_%s_%s_%dx%d.pth" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1])
    model_info = torch.load(model_path)
    model.load_state_dict(model_info["net"])

    output = "checkpoint/%s_%s_%dx%d_%.3f" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1], model_info['acc'])

    if cfg.model.split("-")[0] == "RepVGG":
        # convert to inference module
        model = repvgg_model_convert(model, save_path=output + ".pth")

    model.eval()

    dummy_input1 = torch.randn(1, 3, 128, 128)
    # convert to onnx
    print("convert to onnx......")
    input_names = [ "input"]
    output_names = [ "output" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, output + ".onnx", verbose=False, input_names=input_names, output_names=output_names)
    print("convert done!")

    # convert to libtorch
    with torch.no_grad():
        print("convert to libtorch......")
        traced_script_module = torch.jit.trace(model, dummy_input1)
        traced_script_module.save(output + ".pt")
        print("convert done!")

if __name__ == '__main__':
    main()    

    # convert to MNN
    # ./MNNConvert -f ONNX --modelFile /home/wangjq/wangxt/workspace/pytorch-classifier/checkpoint/best_98.323.onnx --MNNModel /home/wangjq/wangxt/workspace/pytorch-classifier/checkpoint/best_98.323.mnn --bizCode biz

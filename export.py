from __future__ import print_function
from __future__ import division

import os
import torch
import collections
from models.get_network import build_network_by_name, repvgg_model_convert
import config as cfg
import shutil


def main():
    model = build_network_by_name(cfg.model, None, num_classes=len(cfg.classes))

    model_name = "best_%s_%s_%dx%d.pth" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1])
    model_path = os.path.join(cfg.save_checkpoint, model_name)
    model_info = torch.load(model_path)
    model.load_state_dict(model_info["net"])

    output = os.path.join(cfg.save_checkpoint, "%s_%s_%dx%d_%.3f" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1], model_info['acc']))

    if cfg.model.split("-")[0] == "RepVGG":
        # convert to inference module
        model = repvgg_model_convert(model, save_path=output + "_depoly.pth")
        shutil.move(model_path, output + "_train.pth")
    else:
        shutil.move(model_path, output + ".pth")

    model.eval()

    dummy_input1 = torch.randn(1, 3, cfg.input_size[0], cfg.input_size[1])
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

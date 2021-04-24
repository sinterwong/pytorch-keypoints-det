import copy
import torch
from .resnet import resnet10, resnet18, resnet34, resnet50
from .seresnet import se_resnet10, se_resnet18, se_resnet34, se_resnet50
from .repvgg import get_RepVGG_func_by_name
from .mobilenetv3 import mobilenet_v3_small

def build_network_by_name(name, pretrained, num_classes, deploy):
    if name == "resnet10":
        net = resnet10(pretrained=pretrained, num_classes=num_classes)
    elif name == "resnet18":
        net = resnet18(pretrained=pretrained, num_classes=num_classes)
    elif name == "resnet34":
        net = resnet34(pretrained=pretrained, num_classes=num_classes)
    elif name == "resnet50":
        net = resnet50(pretrained=pretrained, num_classes=num_classes)
    elif name == "seresnet10":
        net = se_resnet10(pretrained=pretrained, num_classes=num_classes)
    elif name == "seresnet18":
        net = se_resnet18(pretrained=pretrained, num_classes=num_classes)
    elif name == "seresnet34":
        net = se_resnet34(pretrained=pretrained, num_classes=num_classes)
    elif name == "seresnet50":
        net = se_resnet50(pretrained=pretrained, num_classes=num_classes)
    elif name == "mobilenetv3_small":
        net = mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    elif name.split("-")[0] == "RepVGG":
        repvgg_build_func = get_RepVGG_func_by_name(name)
        net = repvgg_build_func(num_classes=num_classes, pretrained_path=pretrained, deploy=deploy)
    else:
        raise Exception("暂未支持%s network, 请在此处手动添加" % name)
    
    return net

#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example
def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=False):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        state = {
            'net': model.state_dict(),
        }
        torch.save(state, save_path)
    return model

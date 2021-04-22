from .resnet import resnet10, resnet18, resnet34, resnet50
from .seresnet import se_resnet10, se_resnet18, se_resnet34, se_resnet50
from .repvgg import get_RepVGG_func_by_name
from .mobilenetv3 import mobilenet_v3_small

def build_network_by_name(name, pretrained, num_classes):
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
        net = repvgg_build_func(num_classes=num_classes, pretrained_path=pretrained, deploy=False)
    else:
        raise Exception("暂未支持%s network, 请在此处手动添加" % name)
    
    return net


def build_teacher_network_by_name(name, num_classes):
    if name == "resnet18":
        t_net = resnet18(pretrained=None, num_classes=num_classes)
    elif name == "resnet34":
        t_net = resnet34(pretrained=None, num_classes=num_classes)
    elif name == "resnet50":
        t_net = resnet50(pretrained=None, num_classes=num_classes)
    elif name == "seresnet18":
        t_net = se_resnet18(pretrained=None, num_classes=num_classes)
    elif name == "seresnet34":
        t_net = se_resnet34(pretrained=None, num_classes=num_classes)
    elif name == "seresnet50":
        t_net = se_resnet50(pretrained=None, num_classes=num_classes)
    elif name.split("-")[0] == "RepVGG":
        repvgg_build_func = get_RepVGG_func_by_name(name)
        t_net = repvgg_build_func(num_classes=num_classes, pretrained_path=None, deploy=True)
    else:
        raise Exception("暂未支持%s teacher network, 请在此处手动添加" % name)
    
    return t_net

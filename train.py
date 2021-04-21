import os
import config as cfg
os.environ["CUDA_VISIBLE_DEVICES"] =str(cfg.device_ids[0]) if len(cfg.device_ids) == 1 else ",".join(cfg.device_ids)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import argparse
from data.dataset import ImageDataSet
from data.transform import data_transform
from models.resnet import resnet10, resnet18, resnet34, resnet50
from models.seresnet import se_resnet10, se_resnet18, se_resnet34, se_resnet50
from models.repvgg import get_RepVGG_func_by_name
from models.mobilenetv3 import mobilenet_v3_small
from utils import progress_bar
from loss.amsoftmax import AMSoftmax
from loss.distill import KLDivLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='PyTorch LPC Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


# get dataloader
trainset = ImageDataSet(root=cfg.train_root, input_size=cfg.input_size, is_train=True, augments_hyp=cfg.augment_hyp)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


testset = ImageDataSet(root=cfg.val_root, input_size=cfg.input_size, is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# Model
print('==> Building model..')
if cfg.model == "resnet10":
    net = resnet10(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "resnet18":
    net = resnet18(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "resnet34":
    net = resnet34(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "resnet50":
    net = resnet50(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "seresnet10":
    net = se_resnet10(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "seresnet18":
    net = se_resnet18(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "seresnet34":
    net = se_resnet34(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "seresnet50":
    net = se_resnet50(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model == "mobilenetv3_small":
    net = mobilenet_v3_small(pretrained=cfg.pretrained, num_classes=cfg.num_classes)
elif cfg.model.split("-")[0] == "RepVGG":
    repvgg_build_func = get_RepVGG_func_by_name(cfg.model)
    net = repvgg_build_func(num_classes=cfg.num_classes, pretrained_path=cfg.pretrained, deploy=False)
else:
    raise Exception("暂未支持%s network, 请在此处手动添加" % cfg.model)


net = net.to(device)
if device == 'cuda' and len(cfg.device_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=range(len(cfg.device_ids)))
    cudnn.benchmark = True

if cfg.teacher:
    if cfg.teacher == "resnet18":
        t_net = resnet18(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.teacher == "resnet34":
        t_net = resnet34(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.teacher == "resnet50":
        t_net = resnet50(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet18":
        t_net = se_resnet18(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet34":
        t_net = se_resnet34(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.model == "seresnet50":
        t_net = se_resnet50(pretrained=None, num_classes=cfg.num_classes)
    elif cfg.teacher.split("-")[0] == "RepVGG":
        repvgg_build_func = get_RepVGG_func_by_name(cfg.teacher)
        t_net = repvgg_build_func(num_classes=cfg.num_classes, pretrained_path=None, deploy=True)
    else:
        raise Exception("暂未支持%s teacher network, 请在此处手动添加" % cfg.teacher)
else:
    t_net = None

# load teacher
if t_net:
    model_info = torch.load(cfg.teacker_ckpt)
    t_net.load_state_dict(model_info["net"])
    t_net = t_net.to(device)
    t_criterion = nn.MSELoss(reduce=True, reduction="mean")

criterion = nn.MSELoss(reduce=True, reduction="mean")

if cfg.optim == "sgd":
    optimizer = optim.SGD(
                        filter(lambda p: p.requires_grad, net.parameters()), 
                        lr=cfg.lr, 
                        momentum=cfg.momentum, 
                        weight_decay=cfg.weight_decay)
elif cfg.optim == "adam":
    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, net.parameters()), 
                        lr=cfg.lr, 
                        betas=(0.9, 0.99), 
                        weight_decay=cfg.weight_decay)
else:
    raise Exception("暂未支持%s optimizer, 请在此处手动添加" % cfg.optim)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

if cfg.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(cfg.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(cfg.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    lr_scheduler = checkpoint['lr_scheduler']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    acc = 0.0
    total = 0
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if cfg.use_amp:
            with torch.cuda.amp.autocast():  # 自动混合精度 (pytorch1.6之后)
                outputs = net(inputs)
                if t_net:
                    with torch.no_grad():
                        teacher_outputs = t_net(inputs)
                    s_loss = criterion(outputs, targets) * 10.
                    d_loss = t_criterion(outputs, teacher_outputs) * 10.
                    loss = s_loss * (1 - cfg.alpha) + d_loss * cfg.alpha
                else:
                    loss = criterion(outputs, targets) * 10.
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets) * 10.

            if t_net:
                with torch.no_grad():
                    teacher_outputs = t_net(inputs)
                s_loss = criterion(outputs, targets) * 10.
                d_loss = t_criterion(outputs, teacher_outputs) * 10.
                loss = s_loss * (1 - cfg.alpha) + d_loss * cfg.alpha

        if cfg.use_amp:
            # Scales loss. 为了梯度放大.
            scaler.scale(loss).backward()
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        # print(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0).size())
        # print(torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0).size())
        # print((torch.mean(targets, dim=0) - targets).size())
        # print((torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0)).size())
        # acc += 1 - torch.mean((outputs.detach() - targets.detach()) ** 2) / torch.var(targets)
        acc += 1 - torch.mean(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d)'
            % (train_loss/(batch_idx+1), 100.*acc/(batch_idx+1), total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    acc = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets) * 10.
                test_loss += loss.item()
                total += targets.size(0)
                acc += 1 - torch.mean(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0))

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d)'
                % (test_loss/(batch_idx+1), 100.*acc/(batch_idx+1), total))

    # Save checkpoint.
    acc = 100.*acc/(batch_idx+1)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr_scheduler': lr_scheduler,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(cfg.save_checkpoint, "best_%s_%s_%dx%d.pth" % (cfg.model, cfg.data_name, cfg.input_size[0], cfg.input_size[1])))
        best_acc = acc


for epoch in range(start_epoch, start_epoch + cfg.epoch):
    train(epoch)
    lr_scheduler.step()
    test(epoch)

import os
import config as cfg
os.environ["CUDA_VISIBLE_DEVICES"] =str(cfg.device_ids[0]) if len(cfg.device_ids) == 1 else ",".join(cfg.device_ids)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import argparse
from data.dataset import KeypointsDetDataSet
from data.transform import fast_transform, data_aug
from models.get_network import build_network_by_name
from tools.utils import progress_bar
from tools.distill import DistillForFeatures
from loss.awloss import AdaptiveWingLoss
from loss.wloss import WingLoss
from loss.distill import DistillFeatureMSELoss
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
aug_seq = data_aug()
transform = fast_transform()
trainset = KeypointsDetDataSet(root=cfg.train_root, input_size=cfg.input_size, is_train=True, transform=transform, data_aug=aug_seq)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

testset = KeypointsDetDataSet(root=cfg.val_root, input_size=cfg.input_size, is_train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# Model
print('==> Building model..')
net = build_network_by_name(cfg.model, cfg.pretrained, cfg.num_classes, deploy=False)

net = net.to(device)
if device == 'cuda' and len(cfg.device_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=range(len(cfg.device_ids)))
    cudnn.benchmark = True

# Knowledge Distillation 
if cfg.teacher:
    print('==> Building teacher model..')
    t_net = build_network_by_name(cfg.teacher, None, cfg.num_classes, deploy=True)
else:
    t_net = None

# Load teacher weights
if t_net:
    model_info = torch.load(cfg.teacker_ckpt)
    t_net.load_state_dict(model_info["net"])
    t_net = t_net.to(device)
    t_criterion = nn.MSELoss(reduce=True, reduction="mean")

# Get the intermediate output predistill from teacher and student
if t_net and cfg.dis_feature:
    f_distill = DistillForFeatures(cfg.dis_feature, net, t_net)
    fs_criterion = DistillFeatureMSELoss(reduction="mean", num_df=len(cfg.dis_feature))

# criterion = nn.MSELoss(reduce=True, reduction="mean")
# criterion = AdaptiveWingLoss()
criterion = WingLoss()  # 有分段, 对离群点处理会好一点, 适合数据不干净时使用
# criterion = nn.SmoothL1Loss(reduce=True, reduction='sum')

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

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)  # 等步长衰减
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)  # 每步都衰减(γ 一般0.9+)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch // 30)  # 余弦式周期策略

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
    if t_net and cfg.dis_feature:
        hooks = f_distill.get_hooks()

    # 自动混合精度
    scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 自动混合精度 (pytorch1.6之后)
            outputs = net(inputs)
            if t_net:
                loss = torch.cuda.FloatTensor([0]) if outputs.is_cuda else torch.Tensor([0])
                with torch.no_grad():
                    teacher_outputs = t_net(inputs)
                if cfg.dis_feature:
                    t_out = []
                    s_out = []
                    for k, v in f_distill.activation.items():
                        g, k, n = k.split("_")
                        # 一一配对feature, 进行loss 计算
                        if g == "s":
                            s_out.append(v)
                        else:
                            t_out.append(v)
                    # 选定的 feature 分别计算loss
                    fs_loss = fs_criterion(s_out, t_out)
                    loss += fs_loss
                s_loss = criterion(outputs, targets)
                do_loss = t_criterion(outputs, teacher_outputs)
                loss += (s_loss * (1 - cfg.alpha) + do_loss * cfg.alpha)
            else:
                loss = criterion(outputs, targets)
            # Scales loss. 放大梯度.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss += loss.item()
        total += targets.size(0)
        acc += 1 - torch.mean(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0))
        progress_bar(batch_idx, len(trainloader), 'Current lr: %f | Loss: %.5f | Acc: %.3f%% (%d)'
            % (optimizer.state_dict()['param_groups'][0]['lr'], train_loss/(batch_idx+1), 100.*acc/(batch_idx+1), total))
    
    if t_net and cfg.dis_feature:
        for hook in hooks:
            hook.remove()


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
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc += 1 - torch.mean(torch.sum((outputs.detach() - targets.detach()) ** 2, dim=0) / torch.sum((torch.mean(targets, dim=0) - targets) ** 2, dim=0))

            progress_bar(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.3f%% (%d)'
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

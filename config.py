import os

# dataset
input_size = (224, 224)

data_name = "handpose"
num_points = 21
num_classes = num_points * 2
train_root = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/train_val'
val_root = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/train_val'

# solver
device_ids = [2]
batch_size = 64
epoch = 3000
optim = "sgd"
lr_gamma = 0.5  # 衰减比率
lr_step_size = 40  # 多少 epoch 衰减一次
lr = 1e-2
momentum = 0.9
weight_decay = 1e-5
num_workers = 8

# model info
model = "seresnet10"
pretrained = "weights/resnet50-19c8e357.pth"
resume = None

# knowledge distill
teacher = None
teacker_ckpt = "checkpoint/seresnet34_handpose_224x224_86.915.pth"
alpha = 0.01  # 当 alpha 为0时, 意味着不使用 output 进行蒸馏
temperature = 6
# 要求老师网络所选层和学生网络所选层维度一致, 或使用conv1x1 对学生网络进行调整
dis_feature = {
    'layer1': (0, 'bn2'), 
    'layer1': (1, 'bn2'), 
    'layer2': (0, 'bn2'), 
    'layer2': (1, 'bn2'), 
    'layer3': (0, 'bn2'),
    'layer3': (1, 'bn2'),
}

save_checkpoint = os.path.join('checkpoint', data_name, model)
if not os.path.exists(save_checkpoint):
    os.makedirs(save_checkpoint)

# dataset
input_size = (224, 224)

augment_hyp = {
    'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.0,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0, 
    'fliplr' : True,
    'flipud' : True
}
data_name = "handpose"
num_points = 21
num_classes = num_points * 2  # 每个点存在三个属性: x, y, z(该点是否遮挡)
train_root = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/train'
val_root = '/home/wangjq/wangxt/datasets/gesture-dataset/handpose_datasets_v1/val'

# solver
device_ids = [2]
batch_size = 64
epoch = 300
optim = "sgd"
lr_gamma = 0.5  # 衰减比率
lr_step_size = 35  # 多少 epoch 衰减一次
lr = 2e-3
momentum = 0.9
weight_decay = 5e-4
num_workers = 8
use_amp = False

# model info
model = "seresnet18"
pretrained = "weights/resnet18-5c106cde.pth"
save_checkpoint = 'checkpoint'
resume = None

# knowledge distill
# teacher = "resnet50"
teacher = None
teacker_ckpt = "checkpoint/resnet50_handpose_224x224_88.pth"
alpha = 0.9
temperature = 6

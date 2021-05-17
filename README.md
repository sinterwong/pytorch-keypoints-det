# pytorch-pose

基于点回归的关键点检测

## Requirements

- onnx == 1.8.1
- opencv-python == 4.2.0.32
- pytorch == 1.6.0

## Configuration

工程中所有可配置的超参数都在 config.py 中，有五部分（暂时还没有模块化）分别是：

- dataset
  - ps1: 目前使用的是imgaug提供的数据增强pipe；所以数据增强的超参数未生效（生效于 torchvision 提供的数据增强），关于数据增强的代码在 data/transform.py 中。
  - ps2: 数据集目录格式为：
    - train
      - images
      - labels
- solver
- loss（目前还在 train.py 中选择。。。）
- model
- knowledge distill
  - based output 关键参数: teacher & teacher_ckpt & alpha & temperature
  - based feature 关键参数: dis_feature

## Training

在 config.py 中配置好参数后，执行以下命令进行训练

```
python train.py
```

训练时，模型会根据配置信息在 checkepoint 下创建目录结构并命名文件。

## Inference

在 demo.py 中配置完后，执行以下命令对单张图片进行推理

```
python demo.py
```

## Deploy

模型训练完成之后，在 config.py 中 model & data_name和训练时保持一致后，运行以下命令生成 .onnx 和 .pt 文件。

```
python export.py
```




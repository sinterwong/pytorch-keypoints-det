import torchvision.transforms as T
import config as cfg


def data_transform(is_train=True):
    # normalize_transform = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.input_size),
            T.RandomHorizontalFlip(p=cfg.flip_prob),
            T.RandomVerticalFlip(p=cfg.flip_prob),
            T.RandomAffine(cfg.affine_prob),
            # T.RandomErasing(),
            T.RandomRotation(cfg.ro_degree),
            T.ColorJitter(brightness=cfg.bright_prob, saturation=cfg.satura_prob,
                          contrast=cfg.contrast_prob, hue=cfg.hue_prob),
            T.Pad(cfg.pad),
            T.RandomCrop(cfg.input_size),
            T.ToTensor(),
            # normalize_transform,
        ])

    else:
        transform = T.Compose([
            T.Resize(cfg.input_size),
            T.ToTensor(),
            # normalize_transform
        ])

    return transform

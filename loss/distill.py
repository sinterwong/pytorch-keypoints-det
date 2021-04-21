import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss():
    def __init__(self, alpha, T):
        super(KLDivLoss).__init__()
        self.alpha = alpha
        self.T = T
        self.KDLoss = nn.KLDivLoss()

    def __call__(self, outputs, t_outputs, labels):
        return self.KDLoss(F.log_softmax(outputs / self.T, dim=1), F.softmax(t_outputs / self.T, dim=1)) * \
                (self.alpha * self.T * self.T) + \
                F.cross_entropy(outputs, labels) * (1. - self.alpha)


class DistillFeatureMSELoss():
    def __init__(self, reduction="mean", num_df=3):
        super(DistillFeatureMSELoss).__init__()
        self.criterion = [nn.MSELoss(reduction=reduction)] * num_df
        self.activation = None

    def __call__(self):
        assert self.activation is not None, "Please complete the 'hook' first."
        t_out = []
        s_out = []
        for k, v in self.activation.items():
            g, k, n = k.split("_")
            # 一一配对feature, 进行loss 计算
            if g == "s":
                s_out.append(v)
            else:
                t_out.append(v)
        # 选定的 feature 分别计算loss 
        fs_loss = [loss_fn(s_out[i], t_out[i]) for i, loss_fn in enumerate(self.criterion)]
        return torch.sum(torch.cuda.FloatTensor(fs_loss) if v.is_cuda else torch.Tensor(fs_loss)) * 10.

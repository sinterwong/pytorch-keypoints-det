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
    def __init__(self, reduction="mean", num_df=3, alpha=10.):
        super(DistillFeatureMSELoss).__init__()
        self.criterion = [nn.MSELoss(reduction=reduction)] * num_df
        self.alpha = alpha

    def __call__(self, s_out, t_out):
        fs_loss = [loss_fn(s_out[i], t_out[i]) for i, loss_fn in enumerate(self.criterion)]
        return torch.sum(torch.cuda.FloatTensor(fs_loss) if s_out[0].is_cuda else torch.Tensor(fs_loss)) * self.alpha

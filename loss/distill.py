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

class DistillMSELoss():
    def __init__(self):
        super(DistillMSELoss).__init__()
        self.MSELoss = nn.MSELoss(reduction="none")

    def __call__(self, outputs, t_outputs):
        return self.MSELoss(outputs, t_outputs)

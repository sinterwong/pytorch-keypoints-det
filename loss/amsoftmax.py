import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, input, target, scale=10.0, margin=0.35):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        output = output * scale

        logpt = F.log_softmax(output)
        # logpt = F.logsigmoid(output)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss


if __name__ == '__main__':

    print(loss.detach().numpy())
    print(list(criteria.parameters())[0].shape)
    print(type(next(criteria.parameters())))

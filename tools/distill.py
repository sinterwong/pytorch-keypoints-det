import torch.nn as nn

class DistillForFeatures():
    def __init__(self, info, s_net, t_net):
        super(DistillForFeatures).__init__()
        self.info = info
        self.s_net = s_net
        self.t_net = t_net
        self.activation = {}
    
    def get_activation(self, name):
        def hook(model, inputs, outputs):
            self.activation[name] = outputs
        return hook
    
    def get_hooks(self):
        hooks = []
        for k, (idx, name) in self.info.items():
            # S-model
            hooks.append(self.s_net._modules[k][idx]._modules[name].register_forward_hook(self.get_activation("s_{}_{}".format(k, name))))
            # T-model
            hooks.append(self.t_net._modules[k][idx]._modules[name].register_forward_hook(self.get_activation("t_{}_{}".format(k, name))))
        return hooks


import torch.nn as nn
import torch
import numpy as np
import copy
# Auxiliary functions useful for GEM's inner optimization.
class End2EndMPNet(nn.Module):
    def __init__(self, total_input_size, AE_input_size, mlp_input_size, output_size, CAE, MLP):
        super(End2EndMPNet, self).__init__()
        self.encoder = CAE.Encoder()
        self.mlp = MLP(mlp_input_size, output_size)
        self.mse = nn.MSELoss()
        self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        self.total_input_size = total_input_size
        self.AE_input_size = AE_input_size

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr, momentum=momentum)
    def forward(self, x, obs):
        # xobs is the input to encoder
        # x is the input to mlp
        #z = self.encoder(x[:,:self.AE_input_size])
        z = self.encoder(obs)
        mlp_in = torch.cat((z,x), 1)    # keep the first dim the same (# samples)
        return self.mlp(mlp_in)

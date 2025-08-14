import torch
import torch.nn as nn

###############################################################################
# Activation Functions
###############################################################################

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * x.sigmoid()
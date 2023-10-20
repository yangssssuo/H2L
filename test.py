import pandas as pd
import torch
import torch.nn as nn

spr_loss = nn.Spearman_loss()
l1loss = torch.nn.L1Loss(reduction="mean")
aaa = torch.randn(32,1,100).requires_grad_()
bbb = torch.randn(32,1,100)

spr = spr_loss(aaa,bbb)
l1 = l1loss(aaa,bbb)
print(spr)
print(l1)
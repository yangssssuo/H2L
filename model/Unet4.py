import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,n_embd,n_head) -> None:
        super(Attention,self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=n_embd,num_heads=n_head)

    def forward(self,x):
        x = x.permute(2,0,1)
        x,attn_w = self.attention(x,x,x)
        x = x.permute(1,2,0)
        return x,attn_w

class SimAttention(nn.Module):
    def __init__(self,n_embd,n_head) -> None:
        super(SimAttention,self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=n_embd,num_heads=n_head)

    def forward(self,x):
        x = x.permute(2,0,1)
        x,_ = self.attention(x,x,x)
        x = x.permute(1,2,0)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.batchnorm1 = nn.BatchNorm1d(1)
        self.do = nn.Dropout(0.3)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.dense1 = SimAttention(16,8)

        self.batchnorm2 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.dense2 = SimAttention(32,8)

        self.batchnorm3 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.dense3 = SimAttention(64,8)

        self.batchnorm4 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.dense4 = SimAttention(128,8)

        self.batchnorm5 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.attn5 = Attention(256,8)

        self.batchnorm6 = nn.BatchNorm1d(256)
        self.upconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.dense6 = SimAttention(128,8)

        self.batchnorm7 = nn.BatchNorm1d(256)
        self.upconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=64, kernel_size=3, stride=2)
        self.dense7 = SimAttention(64,8)

        self.batchnorm8 = nn.BatchNorm1d(128)
        self.upconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.dense8 = SimAttention(32,8)

        self.batchnorm9 = nn.BatchNorm1d(64)
        self.upconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=3, stride=2)
        self.dense9 = SimAttention(16,8)

        self.batchnorm0 = nn.BatchNorm1d(32)
        self.upconv5 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=4, stride=2)
        self.out_put_lin = nn.Linear(4000,4000)
        # self.out_put_lins = nn.ModuleList(
        #     nn.Linear(1000,1000) for _ in range(3)
        # )

    def forward(self, x):
        x = self.batchnorm1(x)
        x1 = F.relu(self.conv1(x))#out:[32,16,1999]
        x1 = x1 + self.do(self.dense1(x1))
        x1 = self.batchnorm2(x1)
        
        x2 = F.relu(self.conv2(x1))#out:[32,32,999]
        x2 = x2 + self.do(self.dense2(x2))
        x2 = self.batchnorm3(x2)

        x3 = F.relu(self.conv3(x2))#out:[32.64,499]
        x3 = x3 + self.do(self.dense3(x3))
        x4 = self.batchnorm4(x3)

        x4 = F.relu(self.conv4(x3))#out:[32,128,249]
        x4 = x4 + self.do(self.dense4(x4))
        x4 = self.batchnorm5(x4)

        x5 = F.relu(self.conv5(x4))#out:[32,256,124]

        x5,attn_w = self.attn5(x5)
        x5 = self.batchnorm6(x5)
        
        x6 = F.relu(self.upconv1(x5))#out:[32,128,249]
        x6 = x6 + self.do(self.dense6(x6))
        # x6 = self.batchnorm7(x6)

        x7 = torch.cat((x6,x4), dim=1)
        x7 = self.batchnorm7(x7)
        x7 = F.relu(self.upconv2(x7))#out:[32.64,499]
        x7 = x7 + self.do(self.dense7(x7))

        x8 = torch.cat((x7,x3), dim=1)
        x8 = self.batchnorm8(x8)
        x8 = F.relu(self.upconv3(x8))#out:[32,32,999]
        x8 = x8 + self.do(self.dense8(x8))

        x9 = torch.cat((x8,x2), dim=1)
        x9 = self.batchnorm9(x9)
        x9 = F.relu(self.upconv4(x9))
        x9 = x9 + self.do(self.dense9(x9))

        x0 = torch.cat((x9,x1), dim=1)
        x0 = self.batchnorm0(x0)
        x0 = F.relu(self.upconv5(x0))#out:[32,32,999]
        out = self.out_put_lin(x0)
        # for layer in self.out_put_lins:
        #     out = self.do(F.relu(layer(out)))
        return out,attn_w
    
if __name__ == "__main__":
    test_net = UNet()
    # test_attn = Attention(1,1)
    a = torch.randn(32,1,4000)
    # lin = nn.Linear(4000,4000)
    # a_lin = lin(a)
    # print(a_lin.shape)
    b = test_net(a)
    # # c = test_attn(a)
    print(b[0].shape)
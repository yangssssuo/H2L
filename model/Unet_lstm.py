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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.activate = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_s):
        # print('in shape:',x.shape)
        x = self.activate(x)
        x = x.permute(0,2,1)
        lstm_out, h_s = self.lstm(x,h_s)
        outs = []
        for time_step in range(lstm_out.size(1)):
            outs.append(self.fc(lstm_out[:,time_step,:]))
        outs = torch.stack(outs, dim=1)
        outs = outs.permute(0,2,1)
        # print('out shape:',outs.shape)
        # print('h_s0 shape:',h_s[0].shape)
        # print('h_s1 shape:',h_s[1].shape,'\n')
        return outs, h_s

class UnetBlockIn(nn.Module):
    def __init__(self,length,input_dim,hidden_dim,output_dim) -> None:
        super(UnetBlockIn,self).__init__()

        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.lstm = LSTMModel(input_dim,hidden_dim,output_dim)
        self.conv = nn.Conv1d(output_dim,output_dim,kernel_size=3,stride=2)
        self.lin_lenth = self._conv1d_output_length(length)
        self.linear = nn.Linear(self.lin_lenth,self.lin_lenth)

    def _conv1d_output_length(self,input_length, kernel_size=3, stride=2, padding=0, dilation=1):
        return int(((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    
    def forward(self,x,h_s):
        x = self.batchnorm(x)
        x,h_s = self.lstm(x,h_s)
        x = self.conv(x)
        x = self.linear(x)
        return x,h_s
    
class UnetBlockOut(nn.Module):
    def __init__(self,length,input_dim,hidden_dim,output_dim) -> None:
        super(UnetBlockOut,self).__init__()

        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.lstm = LSTMModel(input_dim,hidden_dim,output_dim)
        self.conv = nn.ConvTranspose1d(output_dim,output_dim,kernel_size=3,stride=2)
        self.lin_lenth = self._convtranspose1d_output_length(length)
        self.linear = nn.Linear(self.lin_lenth,self.lin_lenth)

    def _convtranspose1d_output_length(self,input_length, kernel_size=3, stride=2, padding=0, output_padding=0, dilation=1):
        return int((input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1)
    
    def forward(self,x,h_s):
        x = self.batchnorm(x)
        x,h_s = self.lstm(x,h_s)
        x = self.conv(x)
        x = self.linear(x)
        return x,h_s

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UnetBlockIn(4000,1,16,16)
        self.down2 = UnetBlockIn(1999,16,32,32)
        self.down3 = UnetBlockIn(999,32,64,64)
        self.down4 = UnetBlockIn(499,64,128,128)
        self.down5 = UnetBlockIn(249,128,256,256)
        self.attn = Attention(256,8)
        self.up1 = UnetBlockOut(124,256,256,128)
        self.up2 = UnetBlockOut(249,128,128,64)
        self.up3 = UnetBlockOut(499,64,64,32)
        # self.up4 = UnetBlockOut(499,128,128,64)
        self.out_put = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=2),
            nn.Linear(1000,1000),
            nn.Linear(1000,1000),
            nn.Linear(1000,1000),
        ])
        self.oup = nn.Linear(1000,1000)
    
    def forward(self,x):
        x1,h_s1 = self.down1(x,None)
        x2,h_s2 = self.down2(x1,None)
        x3,h_s3 = self.down3(x2,None)
        x4,h_s4 = self.down4(x3,None)
        x5,h_s5 = self.down5(x4,None)

        x5,attn_w = self.attn(x5)

        x6,h_s6 = self.up1(x5,h_s5)
        x7,h_s7 = self.up2(x6 ,h_s4)
        out,h_s8 = self.up3(x7,h_s3)
        for layer in self.out_put:
            out = F.relu(layer(out))
        out = self.oup(out)
        return out,attn_w


if __name__ == '__main__':
    aaa = torch.randn(2,1,4000)
    # lstm = UnetBlockIn(4000,1,3,8)
    model = UNet()
    bbb = model(aaa)
    ccc = torch.randint(0, 8, (4000,))
    print(bbb[0].shape)
    # print(ccc.shape)
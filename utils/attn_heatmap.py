import torch
import matplotlib.pyplot as plt
import seaborn as sns


aaa = torch.load('attn_w/attn_999.pt')
aaa = torch.nn.functional.softmax(aaa)
# print(aaa.shape)
plt.figure(figsize=(20,20),dpi=600)

sns.heatmap(aaa,xticklabels=False, yticklabels=False,square=True,cmap='mako')
plt.savefig(f'test_heat.png')
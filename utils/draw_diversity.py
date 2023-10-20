import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font3 = {'family':'Times New Roman',
         'weight':'semibold',
         'size':15,}

plt.figure(figsize=(20,5),dpi=600)
plt.xlabel(r'Wave Number (cm$^{-1}$)',fontdict=font3)
plt.ylabel('IR (a.u.)',fontdict=font3)
plt.xlim(0,4000)
plt.ylim(0,400)
for idx in range(1,3501): 
    if idx == 1355:pass
    elif idx == 2607:pass
    elif idx == 3039:pass
    else:
        aaa = pd.read_csv(f'/home/yanggk/Data/H2L_Data/IR/ori/curve/{idx}.txt',header=None)
        # bbb = pd.read_csv(f'/home/yanggk/Data/H2L_Data/Raman/ori/curve/{idx}.txt',header=None)
        # print(aaa[1])
        freq1 = aaa[0]
        inti1 = aaa[1]
        plt.plot(freq1,inti1,linewidth=0.8,color='#1B6B93',alpha=0.005)
        plt.fill_between(freq1,inti1,color='#1B6B93',alpha=0.005)
plt.savefig('IR.png')
plt.cla()
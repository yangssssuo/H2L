from email.header import Header
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


for idx in range(1,3501): 
    if idx == 1355:pass
    elif idx == 2607:pass
    elif idx == 3039:pass
    else:
        with open(f'/home/yanggk/Data/HW/IR_Curve/{idx}.txt','r') as f:
            x = []
            y = []
            aaa = f.readlines()
            for i in aaa:
                i = i.replace('\n','').split(' ')
                while '' in i:
                    i.remove('')
                x.append(float(i[0]))
                y.append(float(i[1]))
            x = np.array(x)
            y = np.array(y)

            plt.plot(x,y)
            plt.savefig(f'/home/yanggk/Data/HW/IR_fig/{idx}.png')
            plt.cla()


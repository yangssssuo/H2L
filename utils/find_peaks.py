import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

class Peak_handler():
    def __init__(self,txt_path) -> None:
        self.path = txt_path
        self.idx = txt_path.split('.')[0]
        aaa = pd.read_csv(txt_path,header=None)
        self.x = np.array(aaa[0].values)
        self.y = np.array(aaa[1].values)
        self.peaks, heights = find_peaks(self.y,height=0)
        self.height = heights['peak_heights']
        self.num_peaks = len(self.peaks)
    
    def draw_pict(self):
        plt.plot(self.x,self.y)
        plt.plot(self.peaks,self.height,'x')
        plt.savefig(f'{self.idx}.png')
        plt.cla()
    
    def save_nums(self):
        with open(f'sum.txt','a') as w:
            w.writelines(f'{self.idx},{self.num_peaks}\n')


def cal_peaks(file_path):
    aaa = pd.read_csv(file_path,header=None)
    avg_n = np.array(aaa[1]).mean()
    return avg_n


# if __name__ == '__main__':
#     for idx in range(1,3501): 
#         if idx == 1355:pass
#         elif idx == 2607:pass
#         elif idx == 3039:pass
#         else:
#             xxx = Peak_handler(f'{idx}.txt')
#             xxx.draw_pict()
#             xxx.save_nums()
#             print(f'{idx}/3500 Done')

aaa = cal_peaks('/home/yanggk/Data/HW/Raman/noised/peaks/sum.txt')
bbb = cal_peaks('/home/yanggk/Data/HW/Raman/ori/peaks/sum.txt')
print(aaa)
print(bbb)
ccc = cal_peaks('/home/yanggk/Data/HW/IR/noised/peaks/sum.txt')
ddd = cal_peaks('/home/yanggk/Data/HW/IR/ori/peaks/sum.txt')
print(ccc)
print(ddd)

    






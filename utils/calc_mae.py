import pandas as pd
import torch
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import math
# import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,spearmanr,wasserstein_distance
import numpy as np
# y_true = torch.tensor([3, -0.5, 2, 7])
# y_pred = torch.tensor([2.5, 0.0, 2, 8])

# mae = torch.mean(torch.abs(y_true - y_pred))
# mse = torch.mean((y_true - y_pred) ** 2)
# rmse = torch.sqrt(mse)
# r2 = r2_score(y_true.numpy(), y_pred.numpy())

def mae(a,b):
    return mean_absolute_error(a,b)

def mse(a,b):
    return mean_squared_error(a,b)

def rmse(a,b):
    return math.sqrt(mse(a,b))

def r2(a,b):
    return r2_score(a,b)

def relative_err(a,b):
    '''a是真值'''
    r_errs = [abs(x-y) / x for x,y in zip(a,b)]
    mean_r_err = sum(r_errs) / len(r_errs)
    return mean_r_err

def earth_mover(a,b):
    return wasserstein_distance(a,b)

def spearman(a,b):
    return spearmanr(a,b)[0]
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2 score: {r2}')
def error_jud(a,b,i):
    if i == 0 :
        return mae(a,b)
    elif i == 1:
        return mse(a,b)
    elif i == 2:
        return rmse(a,b)
    elif i ==3:
        return relative_err(a,b)
    elif i == 4:
        return r2(a,b)
    elif i ==5:
        return spearman(a,b)
    elif i ==6:
        return earth_mover(a,b)

def density(a):
    dens = gaussian_kde(a)
    xs = np.linspace(0,1,1000)
    ys = dens(xs)
    xs = xs.tolist()
    ys = ys.tolist()
    return xs,ys

ori_lis = []
pred_lis = []

for i in range(350):
    aaa = pd.read_csv(f'spec_data/{i}.csv')
    # print(aaa)
    # print(aaa['ori'])
    # print(aaa['noi_inp'])
    # print(aaa['output'])
    ori = aaa['ori'].values
    # print(ori)
    noi_inp = aaa['noi_inp'].values
    output = aaa['output'].values
    
    ori_noi = error_jud(ori,noi_inp,5)
    ori_pred = error_jud(ori,output,5)
    ori_lis.append(ori_noi)
    pred_lis.append(ori_pred)
    print(i,'noi:',ori_noi,'pred:',ori_pred)
plt.figure(dpi=600)
# plt.xlim(0,500)
plt.style.use('seaborn-paper')
plt.hist(ori_lis,histtype='bar',bins=20,stacked=True,density=True,label='Raw & Clear Spectra',color='#82B0D2')
plt.hist(pred_lis,histtype='bar',bins=20,stacked=True,density=True,label='Processed & Clear Spectra',color='#FFBE7A')
xs1,ys1 = density(ori_lis)
xs2,ys2 = density(pred_lis)

max1 = ys1.index(max(ys1))
max2 = ys2.index(max(ys2))

max_x1,max_y1 = round(xs1[max1],2),round(ys1[max1],2)
max_x2,max_y2 = round(xs2[max2],2),round(ys2[max2],2)


# plt.yticks([])
plt.plot(xs1,ys1,label='Smoothed Raw & Clear',color='#2878b5')
plt.plot(xs2,ys2,label="Smoothed Processed & Clear",color='#FA7F6F')

plt.annotate(f'Average:{max_x1}',(max_x1,max_y1),xytext=(max_x1,max_y1+0.05))
plt.annotate(f'Average:{max_x2}',(max_x2,max_y2),xytext=(max_x2,max_y2-0.02))

plt.ylabel('Probability Density')
plt.xlabel('Distribution of Raman Spearman')
plt.legend()
plt.savefig('hist.png')
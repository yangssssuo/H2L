import pandas as pd
import torch
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import math
# import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,spearmanr,wasserstein_distance
import numpy as np


class Error_Handler():
    def __init__(self,data_path) -> None:
        aaa = pd.read_csv(data_path)
        self.ori = aaa['ori'].values
        self.noi = aaa['noi_inp'].values
        self.pred = aaa['output'].values

    def get_data(self):
        mae = [self._mae(self.ori,self.noi),self._mae(self.ori,self.pred)]
        rmse = [self._rmse(self.ori,self.noi),self._rmse(self.ori,self.pred)]
        erm = [self._earth_mover(self.ori,self.noi),self._earth_mover(self.ori,self.pred)]
        with open('analy.csv','a') as f:
            f.write()

    def err_iter(self,i):
        pass

    def _mae(self,a,b):
        return mean_absolute_error(a,b)

    def _mse(self,a,b):
        return mean_squared_error(a,b)

    def _rmse(self,a,b):
        return math.sqrt(self._mse(a,b))

    def _r2(self,a,b):
        return r2_score(a,b)

    def _relative_err(self,a,b):
        '''a是真值'''
        r_errs = [abs(x-y) / x for x,y in zip(a,b)]
        mean_r_err = sum(r_errs) / len(r_errs)
        return mean_r_err

    def _earth_mover(self,a,b):
        return wasserstein_distance(a,b)

    def _spearman(self,a,b):
        return spearmanr(a,b)[0]
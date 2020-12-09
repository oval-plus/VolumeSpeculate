# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from volume_look.util.utility import Utility
from volume_look.util.save_util import SaveUtil
from volume_look.analysis.build_data import BuildData

class PairAna(object):
    def __init__(self):
        self.t = 90
    
    def get_var(self, k):
        path = '/home/lky/volume_speculate/output/pair_trading/coef/20200102_20200601_{k}.csv'.format_map(vars())
        df = pd.read_csv(path, index_col = 0)
        var = df['coef'].var()
    
    def arma(self):
        whole_df = pd.DataFrame()
        for k in range(0, 3):
            path = '/home/lky/volume_speculate/output/pair_trading/coef/20200102_20200601_{k}.csv'.format_map(vars())
            df = pd.read_csv(path, index_col = 0)
            whole_df = whole_df.append(df)
    
    

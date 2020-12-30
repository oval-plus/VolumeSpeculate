# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import cufflinks as cf
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from cachetools import TTLCache, cached
from volume_look.util.utility import Utility
from volume_look.util.save_util import SaveUtil
from volume_look.analysis.build_data import BuildData
cf.go_offline()
plotly.offline.init_notebook_mode(connected = True)

class Maluyao(object):
    def __init__(self, morning):
        self.morning = morning

    def get_data(self, util, date, k):
        data = BuildData().cal_remain_maturity(util, date)
        data['datetime'] = data['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
        ticker_lst = data['ticker'].unique()
        ticker_1 = ticker_lst[k]
        start_time = util.get_open_time(date, self.morning)
        df = data[data['ticker'] == ticker_1]
        df = df[df['sod'] >= start_time]
        return df

    def cal_spread(self, util, date, k = 0):
        futures_1 = self.get_data(util, date, k)
        futures_2 = self.get_data(util, date, k + 1)

        trade_1 = futures_1['cp'].iloc[0]
        trade_2 = futures_2['cp'].iloc[0]

        # ln_1 = futures_1['cp'].apply(lambda x: np.log(x / trade_1))
        # ln_2 = futures_2['cp'].apply(lambda x: np.log(x / trade_2))
        ln_1 = futures_1['cp'].apply(lambda x: np.log(x))
        ln_2 = futures_2['cp'].apply(lambda x: np.log(x))
        
        ln_1.index = futures_1['datetime'].apply(lambda x: x[11:])
        ln_2.index = futures_2['datetime'].apply(lambda x: x[11:])

        spread = ln_1 - ln_2
        check_date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        spread_df = pd.DataFrame()
        spread_df['spread'] = spread
        spread_df['datetime'] = spread.index.map(lambda x: check_date + " " + x)
        # spread_df['datetime'] = futures_1['datetime']
        spread_df = spread_df.set_index(['datetime'])
        # spread_df.index = futures_1.index
        return spread_df

    def mean_reversion_speed(self, futures):
        cp = futures['cp']
        cp_lst = futures['cp'].shift()

        df = pd.DataFrame()
        df['y'] = cp_lst
        df['x'] = cp
        df = df.dropna(axis = 0, how = 'any')

        model = sm.OLS(df['y'], df['x']).fit()
        params = model.params['x']
        return params

    def get_drift_function(self, util, data):
        drift_df = pd.DataFrame()
        moving_interval = util.get_moving_interval()
        # data = self.get_data(util, date, morning, k)
        drift_func = data['cp'].rolling(moving_interval).mean()
        drift_df['drift'] = drift_func
        drift_df['datetime'] = data['datetime']
        return drift_df

    def get_deriv(self, util, data):
        drift_func = self.get_drift_function(util, data).dropna()
        g0 = drift_func['drift'].shift()
        deriv_func = drift_func['drift'] - g0
        t1 = pd.to_datetime(drift_func['datetime'])
        t2 = pd.to_datetime(drift_func['datetime'].shift())
        gap = (t1 - t2).apply(lambda x: x.total_seconds())
        # gap = pd.Series(deriv_func.reset_index().index.tolist())
        # gap.index = deriv_func.index
        deriv_func = deriv_func * gap
        return deriv_func

    def get_mut(self, util, data):
        speed = self.mean_reversion_speed(data)
        deriv_func = self.get_deriv(util, data)
        lambda_df = self.cal_lambda(util, data)['lambda']
        # drift_func = self.get_drift_function(util, data)
        # mu = 1 / speed * deriv_func + drift_func['drift']
        mu = 1 / speed * deriv_func + lambda_df
        # mu = 1 / speed * deriv_func

        df = pd.DataFrame()
        df['mu'] = mu
        df['datetime'] = data['datetime']
        return df

    def get_std(self, util, data):
        moving_interval = util.get_moving_interval()
        std = data['cp'].rolling(moving_interval).std()

        df = pd.DataFrame()
        df['std'] = std
        df['datetime'] = data['datetime']
        return df
    
    def cal_lambda(self, util, data):
        pct_chg = self.cal_pct_chg(data)
        df = pd.DataFrame()
        df['lambda'] = (pct_chg['pct_chg'] - 0.03)
        df['pct_chg'] = pct_chg['pct_chg'].apply(lambda x: x if x != 0 else 1)
        df['lambda'] = df['lambda'] / df['pct_chg']
        df['datetime'] = data['datetime']
        return df
    
    def cal_interest_rate(self, util, date, k = 0):
        futures_1 = self.get_data(util, date, k)
        futures_2 = self.get_data(util, date, k + 1)
        r = futures_1 / futures_2
        return r

    def cal_pct_chg(self, main_data):
        price_chg_df = pd.DataFrame()
        price_change = main_data['cp'] - main_data['cp'].shift()
        price_chg_df['pct_chg'] = price_change.fillna(0)
        price_chg_df['datetime'] = main_data['datetime']
        return price_chg_df


if __name__ == "__main__":
    
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    util = Utility(config)
    moving_interval = 1200
    date = '20190102'
    k = 0
    morning = True
    

    mly = Maluyao(morning)
    # spread = mly.cal_spread(util, date, k)
    # # theta = mly.mean_reversion_speed(util, date, k)
    # # deriv_func = mly.get_deriv(util, date, k)
    # data = mly.get_data(util, date, k)
    # mu = mly.get_mut(util, data)
    # print(mu)

    r = mly.cal_interest_rate(util, date)
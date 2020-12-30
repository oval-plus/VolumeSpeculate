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


config_path = "/home/lky/volume_speculate/volume_look/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

def get_effective_price(main_data):
    eff_price = (main_data['s1'] * main_data['bv1'] + 
                main_data['b1'] * main_data['sv1']) / (
                main_data['bv1'] + main_data['sv1'])
    return eff_price

def cal_dp(df):
    dp = (df['cp'] - df['cp'].shift()) / df['cp'].shift()
    return dp.dropna()

def get_price_change(main_data):
    price_chg_df = pd.DataFrame()
    price_change = main_data['cp'] - main_data['cp'].shift()
    price_chg_df['pct_chg'] = price_change
    price_chg_df['datetime'] = main_data['datetime']
    return price_chg_df

def data_lag(exog, title, n = 5):
    df = pd.DataFrame()
    for i in range(1, n):
        name = title + '_' + str(i)
        df[name] = exog.shift(i)
    return df

def get_linear_model(independent_df, label):
    independent_df['y'] = label['pct_chg']
    df = independent_df.dropna(axis = 0, how = 'any')
    X = df.drop(['y'], axis = 1)
    X = sm.add_constant(X)
    model = sm.OLS(df['y'], X).fit()
    return model

def trigger(start_date, end_date, name, k, n = 5):
    if name == "eff_price":
        func = get_effective_price
    elif name == "dp":
        func = cal_dp
    util = Utility(config)
    date_lst = util.generate_date_lst(start_date, end_date)
    params_df = pd.DataFrame()
    for i in range(0, len(date_lst)):
        date = date_lst[i]
        data = BuildData().cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        ticker = ticker_lst[k]
        futures = data[data['ticker'] == ticker]
        price_change = get_price_change(futures)
        effective_price = func(futures)
        independent_df = data_lag(effective_price, name, n)
        coefs = get_linear_model(independent_df, price_change).params
        params_df[date] = coefs
    params_df.to_csv('/home/lky/volume_speculate/output/pair_trading/linear/coefs/{name}_{start_date}_{end_date}.csv'.format_map(vars()))


start_date = '20200103'
end_date = '20200303'
# name = 'dp'
name = 'eff_price'
k = 0

trigger(start_date, end_date, name, k, 6)
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

config_path = "/home/lky/volume_speculate/volume_look/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

def get_futures(data, ticker, idx):
    futures = data[data['ticker'] == ticker]
    futures = futures[~futures.index.duplicated()]
    futures = futures.resample('1T', on = 'datetime').first()
    futures = futures.reindex(index = idx)
    return futures

def get_division_ratio(futures_1, futures_2):
    df = pd.DataFrame()
    df['ratio'] = futures_1 / futures_2
    df['ratio'] = df['ratio'].apply(lambda x: np.log(x))
    return df

start_date = '20200103'
end_date = '20200303'

k = 0

util = Utility(config)
date_lst = util.generate_date_lst(start_date, end_date)
for i in range(0, len(date_lst)):
    date = date_lst[i]
    stock_price = util.get_stock_price(date)
    stock_price = stock_price.rename(columns = {'close': "cp"})
    futures = BuildData().cal_remain_maturity(util, date)
    futures['datetime'] = futures['datetime'].apply(
        lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
    
    idx = stock_price.index
    ticker_lst = futures['ticker'].unique()
    ln_df = pd.DataFrame()
    ln_df['datetime'] = stock_price.index
    ln_df.index = ln_df['datetime']
    ln_df['spot'] = stock_price['cp']

    for ticker in ticker_lst:
        futures_df = get_futures(futures, ticker, idx)
        ln_df[ticker] = futures_df['cp']
        ln_df[ticker + 'remain_secs'] = futures_df['remain_secs']
    
    for j in range(0, len(ticker_lst) - 1):
        ticker_close = ticker_lst[j]
        ticker_far = ticker_lst[j + 1]
        ln_ratio = get_division_ratio(ln_df[ticker_close], ln_df[ticker_far])
        ln_df[ticker_close] = ln_ratio['ratio']
    ln_df.to_csv('/home/lky/volume_speculate/output/pair_trading/ln_futures/' + date + '.csv')


    # "ln(futures)"
    # "linear regression on p(t + 1) - p(t) = alpha imbalance(t) + beta imbalance(t - 1) + sigma imbalance(t - 2)"

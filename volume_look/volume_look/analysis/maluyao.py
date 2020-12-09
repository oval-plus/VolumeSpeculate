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

util = Utility(config)


def cal_spread(util, date, morning, k = 0):
    data = BuildData().cal_remain_maturity(util, date)
    data['datetime'] = data['datetime'].apply(
        lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
    ticker_lst = data['ticker'].unique()
    ticker_1 = ticker_lst[k]
    ticker_2 = ticker_lst[k + 1]

    futures_1 = data[data['ticker'] == ticker_1]
    futures_2 = data[data['ticker'] == ticker_2]

    start_time = util.get_open_time(date, morning)
    trade_1 = futures_1[futures_1['sod'] >= start_time]['cp'].iloc[0]
    trade_2 = futures_2[futures_2["sod"] >= start_time]['cp'].iloc[0]

    ln_1 = futures_1['cp'].apply(lambda x: np.log(x / trade_2))
    ln_2 = futures_2['cp'].apply(lambda x: np.log(x / trade_1))
    
    ln_1.index = futures_1['datetime'].apply(lambda x: x[11:])
    ln_2.index = futures_2['datetime'].apply(lambda x: x[11:])

    spread = ln_1 - ln_2
    return spread


date = '20190102'
spread = cal_spread(util, date, True)
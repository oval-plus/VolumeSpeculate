# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from scipy import optimize
from functools import partial
from volume_look.util.utility import Utility

config_path = r"..."
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)
start_date = '20191001'
end_date = '20200201'
date_lst = util.generate_date_lst(start_date, end_date)

n = len(date_lst)
whole_df = pd.DataFrame()
for i in range(0, n):
    date = date_lst[i]
    data = util.open_data(date)
    ticker = data[~data['ticker'].duplicated()]['ticker'].iloc[0]
    data = data[data['ticker'] == ticker]
    data['datetime'] = data['datetime'].apply(
        lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

    data = data[~data.index.duplicated()]
    data['std'] = data['cp'].rolling(window = 10).std()
    data['mean'] = data['cp'].rolling(window = 10).mean()
    data['mean+std'] = data['mean'] + 2 * data['std']
    data['mean-std'] = data['mean'] - 2 * data['std']

    data['buy'] = np.nan
    buy_idx = data[data['mean+std'] < data['cp']]['buy'].index
    data['buy'].loc[buy_idx] = 1

    data['sell'] = np.nan
    sell_idx = data[data['mean-std'] > data['cp']]['sell'].index
    data['sell'].loc[sell_idx] = -1
    
    flag = -1
    for i in range(0, len(data)):
        if data['buy'].iloc[i] == 1 and flag == 1:
            data['buy'].iloc[i] = np.nan
            continue
        if data['sell'].iloc[i] == -1 and flag == -1:
            data['sell'].iloc[i] = np.nan
            continue
        if data['buy'].iloc[i] == 1 and flag == -1:
            flag = 1
            continue
        if data['sell'].iloc[i] == -1 and flag == 1:
            flag = -1
            continue
    sell_n = data['sell'].sum()
    buy_n = data['buy'].sum()

    data['sell'].iloc[-1] = -abs(buy_n + sell_n)

    sell_idx = data['sell'].dropna().index
    buy_idx = data['buy'].dropna().index
    data['commision_sell'] = np.nan
    data['commision_buy'] = np.nan
    data['commision_sell'].loc[sell_idx] = data['cp'].loc[sell_idx] * 0.000345
    data['commision_buy'].loc[buy_idx] = data['cp'].loc[buy_idx] * 0.000345

    data['commision_sell'].iloc[-1] = data['cp'].iloc[-1] * data['sell'].iloc[-1] * 0.000345

    portfolio_df = pd.DataFrame(index = data.index, columns = ['sell'])
    portfolio_df['sell'] = data['sell'] * data['cp']
    portfolio_df['buy'] = data['buy'] * data['cp']
    portfolio_df['sell_vol'] = data['sell']
    portfolio_df['buy_vol'] = data['buy']
    portfolio_df['operation'] = portfolio_df['sell'].fillna(portfolio_df['buy'])
    portfolio_df['commision_sell'] = data['commision_sell']
    portfolio_df['commision_buy'] = data['commision_buy']
    portfolio_df['commision'] = portfolio_df['commision_sell'].fillna(portfolio_df['commision_buy'])
    # portfolio_df['cp'] = (portfolio_df['operation'] - portfolio_df['commision']).cumsum()
    # portfolio_df['cp'] = (portfolio_df['operation']).cumsum()
    # real_pnl = portfolio_df['cp'].reindex(index = portfolio_df['sell'].dropna().index)
    # portfolio_df['pnl'] = real_pnl.reindex(portfolio_df.index)
    portfolio_df = portfolio_df.dropna(axis = 0, how = 'all')
    portfolio_df['datetime'] = data['datetime']

    whole_df = whole_df.append(portfolio_df)
# print(portfolio_df['pnl'].sum())

whole_df = whole_df.set_index(['datetime'])
whole_df['cp'] = (whole_df['operation'] - whole_df['commision']).cumsum()
# whole_df['cp'] = whole_df['operation'].cumsum()
real_pnl = whole_df['cp'].reindex(index = whole_df['sell'].dropna().index)
whole_df['pnl'] = real_pnl.reindex(whole_df.index)

plt.plot(whole_df['pnl'].values)

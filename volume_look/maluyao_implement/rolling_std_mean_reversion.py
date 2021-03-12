# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import calendar
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from scipy import optimize
from functools import partial
from volume_look.util.utility import Utility

config_path = r"D:\Document\SJTU\thesis\program\volume_look\config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)
start_date = '20190423'
# end_date = '20200701'
end_date = '20190601'
date_lst = util.generate_date_lst(start_date, end_date)

def cal_main_data(data, t = 0):
    ticker = data[~data['ticker'].duplicated()]['ticker'].iloc[t]
    data = data[data['ticker'] == ticker]
    data['datetime'] = data['datetime'].apply(
        lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

    data = data[~data.index.duplicated()]
    data['std'] = data['cp'].rolling(window = 100).std()
    data['mean'] = data['cp'].rolling(window = 100).mean()
    data['mean+std'] = data['mean'] + k * data['std']
    data['mean-std'] = data['mean'] - k * data['std']

    data['buy'] = np.nan
    buy_idx = data[data['mean+std'] < data['cp']]['buy'].index
    data['buy'].loc[buy_idx] = 1

    data['sell'] = np.nan
    sell_idx = data[data['mean-std'] > data['cp']]['sell'].index
    data['sell'].loc[sell_idx] = -1

    return data

def cal_portfolio_df(data):
    # sell_n = data['sell'].sum()
    # buy_n = data['buy'].sum()
    # if data['buy'].iloc[-1].isnull():
    #     data['sell'].iloc[-1] = -abs(buy_n + sell_n) if -abs(buy_n + sell_n) != 0 else data['sell'].iloc[-1]
    # else:
    #     data['sell'].iloc[-1] = -abs(buy_n + sell_n - 1) if -abs(buy_n + sell_n - 1) != 0 else data['sell'].iloc[-1]
    #     data['buy'].iloc[-1] = np.nan
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

    portfolio_df = portfolio_df.dropna(axis = 0, how = 'all')
    portfolio_df['datetime'] = data['datetime']
    return portfolio_df

def generate_date_range(date):
    date_range = pd.date_range(start = date + ' 09:30:00', end = date + ' 11:30:00', freq = "0.5S")
    date_range_n = pd.date_range(start = date + ' 13:00:00', end = date + ' 15:00:00', freq = "0.5S")
    date_range = date_range.append(date_range_n)
    date_range = date_range.map(lambda x: str(x))
    date_range = date_range.map(lambda x: x + ".000" if x[-7] != '.' else x[:-3])
    return date_range

def generate_pnl_daterange(date_lst):
    df = pd.Series()
    for i in range(0, len(date_lst)):
        date = date_lst[i]
        date_range = pd.date_range(start = date + ' 09:30:00', end = date + ' 11:30:00', freq = "0.5S")
        date_range_n = pd.date_range(start = date + ' 13:00:00', end = date + ' 15:00:00', freq = "0.5S")
        date_range = date_range.append(date_range_n)
        date_range = date_range.map(lambda x: str(x))
        date_range = date_range.map(lambda x: x + ".000" if x[-7] != '.' else x[:-3])
        df = df.append(pd.Series(date_range))
    return df

def get_date_maturity(util, date):
    year, month = date[0: 4], date[4: 6]
    prefix = year + month
    monthrange = calendar.monthrange(int(year), int(month))[1]
    start_date, end_date = prefix + '01', prefix + str(monthrange)
    maturity_df = util.get_expiration_date(start_date, end_date)
    maturity_date = maturity_df[maturity_df['res'] == True].index[0]
    return maturity_date

def cal_pnl(date_lst, k):
    n = len(date_lst)
    whole_df = pd.DataFrame()
    day_turn = 1
    before_buy, before_sell = 0, 0
    for i in range(0, n):
        date = date_lst[i]
        print(date)
        data = util.open_data(date)
    
        if data.empty:
            continue
        date_range = generate_date_range(date)
        ticker_lst = data[~data['ticker'].duplicated()]['ticker']

        main_data = data[data['ticker'] == ticker_lst.iloc[0]]
        main_data['datetime'] = main_data['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

        main_data = main_data[~main_data['datetime'].duplicated()]
        main_data.index = main_data['datetime']
        main_data = main_data.reindex(index = date_range).fillna(method = 'ffill')

        far_data = data[data['ticker'] == ticker_lst.iloc[1]]
        far_data['datetime'] = far_data['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

        far_data = far_data[~far_data['datetime'].duplicated()]        
        far_data.index = far_data['datetime']
        far_data = far_data.reindex(index = date_range).fillna(method = 'ffill')

        check_df = pd.DataFrame(index = main_data.index)
        check_df['datetime'] = main_data.index
        check_df['cp'] = main_data['cp']
        check_df['cp_far'] = far_data['cp']
        check_df['spread'] = check_df['cp'] - check_df['cp_far']

        check_df['std'] = check_df['spread'].rolling(window = 1000).std()
        check_df['mean'] = check_df['spread'].rolling(window = 1000).mean()
        check_df['mean+std'] = check_df['mean'] + k * check_df['std']
        check_df['mean-std'] = check_df['mean'] - k * check_df['std']

        if day_turn == 1:
            check_df['buy'] = np.nan
            buy_idx = check_df[check_df['mean+std'] < check_df['spread']]['buy'].index[:20]
            check_df['buy'].loc[buy_idx] = 1
            check_df['sell'] = np.nan
            sell_n = before_buy
            buy_n = check_df['buy'].sum() + before_sell

        if day_turn == -1:
            check_df['sell'] = np.nan
            sell_idx = check_df[check_df['mean-std'] > check_df['spread']]['sell'].index[:20]
            check_df['sell'].loc[sell_idx] = -1
            check_df['buy'] = np.nan
            sell_n = check_df['sell'].sum() + before_buy
            buy_n = before_sell
        
        if date == get_date_maturity(util, date):
            if day_turn == 1:
                check_df['buy'] = np.nan
            else:
                check_df['sell'].iloc[-1] = sell_n
            before_buy, before_sell = 0, 0
            operation_df = cal_portfolio_df(check_df)
            operation_df['day_turn'] = day_turn
            day_turn = 1
            whole_df = whole_df.append(operation_df)

            continue

        operation_df = cal_portfolio_df(check_df)
        operation_df['date'] = date
        operation_df['day_turn'] = day_turn
        day_turn = -day_turn
        before_sell = sell_n
        before_buy = buy_n

        whole_df = whole_df.append(operation_df)
    return whole_df
# print(portfolio_df['pnl'].sum())

k = 2
whole_df = cal_pnl(date_lst, k)
whole_df['cp'] = (whole_df['operation'] - whole_df['commision']).cumsum()


pnl_df = pd.read_csv(r'D:\Document\SJTU\thesis\program\output\pnl.csv', index_col = 0)

pnl_df = pd.DataFrame()
for i in range(0, len(date_lst)):
    date = date_lst[i]
    data = util.open_data(date)
    if data.empty:
        continue
    temp_df = pd.DataFrame()
    temp_df['cp'] = data['cp']
    temp_df['date'] = date
    temp_df['datetime'] = data['datetime']
    
    pnl_df = pnl_df.append(temp_df)

pnl_df['datetime'] = pnl_df['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
pnl_df = pnl_df[~pnl_df['datetime'].duplicated()]
pnl_df.index = pnl_df['datetime']
mkt_pnl = pnl_df['diff']
mkt_pnl.index = pd.to_datetime(pnl_df['datetime'])

fig, ax = plt.subplots()
ax.plot(real_pnl)
ax.plot(mkt_pnl.cumsum())
fig.autofmt_xdate()
plt.title("20190423-20200701 IF (no commission fee)")
plt.savefig(r'D:\Document\SJTU\thesis\program\output\figure\rolling_std_04230701')

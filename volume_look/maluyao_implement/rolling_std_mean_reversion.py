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


# 1minute
config_path = r"..."
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)
start_date = '20190423'
# start_date = '20190510'
end_date = '20200701'
# end_date = '20190701'
date_lst = util.generate_date_lst(start_date, end_date)

def cal_portfolio_df(data):
    sell_idx = data['sell'].dropna().index
    buy_idx = data['buy'].dropna().index
    long_idx = data['long'].dropna().index
    short_idx = data['short'].dropna().index
    data['commision_sell'] = np.nan
    data['commision_buy'] = np.nan
    data['commision_short'] = np.nan
    data['commision_long'] = np.nan
    data['commision_sell'].loc[sell_idx] = data['cp'].loc[sell_idx] * 0.000023
    data['commision_buy'].loc[buy_idx] = data['cp'].loc[buy_idx] * 0.000023
    data['commision_long'].loc[long_idx] = data['cp_far'].loc[long_idx] * 0.000023
    data['commision_short'].loc[short_idx] = data['cp_far'].loc[short_idx] * 0.000023

    data['commision_sell'].iloc[-1] = data['cp'].iloc[-1] * data['sell'].iloc[-1] * 0.000023
    data['commision_short'].iloc[-1] = data['cp_far'].iloc[-1] * data['short'].iloc[-1] * 0.000023

    portfolio_df = pd.DataFrame(index = data.index, columns = ['sell'])
    portfolio_df['sell'] = data['sell'] * data['cp']
    portfolio_df['buy'] = data['buy'] * data['cp']
    portfolio_df['short'] = data['short'] * data['cp_far']
    portfolio_df['long'] = data['long'] * data['cp_far']
    portfolio_df['sell_vol'] = data['sell']
    portfolio_df['buy_vol'] = data['buy']
    portfolio_df['short_vol'] = data['short']
    portfolio_df['long_vol'] = data['long']
    portfolio_df['operation'] = portfolio_df['sell'].fillna(portfolio_df['buy'])
    portfolio_df["operation_far"] = portfolio_df['short'].fillna(portfolio_df['long'])
    portfolio_df['commision_sell'] = data['commision_sell']
    portfolio_df['commision_buy'] = data['commision_buy']
    portfolio_df["commision_long"] = data['commision_long']
    portfolio_df["commision_short"] = data['commision_short']
    portfolio_df['commision'] = portfolio_df['commision_sell'].fillna(portfolio_df['commision_buy'])
    portfolio_df['commision_far'] = portfolio_df["commision_short"].fillna(portfolio_df["commision_long"])

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

def generate_operate_idx(idx):
    before = idx.iloc[0]
    new_idx = [idx.index[0]]
    for i in range(1, len(idx)):
        bi = idx.iloc[i]
        if int(bi - before) >= 120:
            new_idx.append(idx.index[i])
            before = idx.iloc[i]
    return new_idx

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

        check_df['std'] = check_df['spread'].rolling(window = 1200).std()
        check_df['mean'] = check_df['spread'].rolling(window = 1200).mean()
        check_df['mean+std'] = check_df['mean'] + k * check_df['std']
        check_df['mean-std'] = check_df['mean'] - k * check_df['std']
        check_df['interval'] = [i for i in range(1, len(check_df) + 1)]

        if day_turn == 1:
            check_df['buy'] = np.nan
            check_df['short'] = np.nan
            _buy_idx = check_df[check_df['mean+std'] < check_df['spread']]["interval"]
            times = 20 if (before_sell == -20 or before_sell == 0) else -int(before_sell)
            buy_idx = generate_operate_idx(_buy_idx)[:times]

            check_df['buy'].loc[buy_idx] = 1
            check_df["short"].loc[buy_idx] = -1
            check_df['sell'] = np.nan
            check_df['long'] = np.nan
            
            short_n = check_df['short'].sum() + (-before_sell)
            long_n = before_buy

            buy_n = check_df['buy'].sum() + before_sell
            sell_n = before_buy

        if day_turn == -1:
            check_df['sell'] = np.nan
            check_df['long'] = np.nan
            _sell_idx = check_df[check_df['mean-std'] > check_df['spread']]["interval"]
            times = 20 if (before_buy == 20 or before_buy == 0) else int(before_buy)
            sell_idx = generate_operate_idx(_sell_idx)[:times]

            check_df['sell'].loc[sell_idx] = -1
            check_df['long'].loc[sell_idx] = 1
            check_df['buy'] = np.nan
            check_df["short"] = np.nan
            sell_n = check_df['sell'].sum() + before_buy
            long_n = check_df["long"].sum() + (-before_buy)
            buy_n = before_sell
            short_n = before_sell
        
        if date == get_date_maturity(util, date):
            if day_turn == 1:
                check_df['buy'] = np.nan
                check_df['short'] = np.nan
            else:
                check_df['sell'].iloc[-1] = sell_n
                check_df['long'].iloc[-1] = long_n
            before_buy, before_sell = 0, 0
            operation_df = cal_portfolio_df(check_df)
            operation_df['day_turn'] = day_turn
            operation_df['date'] = date
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

k = 2.5
whole_df = cal_pnl(date_lst, k)
whole_df['cp'] = (whole_df['operation'] + whole_df['operation_far'] - 
            whole_df['commision'] - whole_df["commision_far"]).cumsum()
# whole_df['operation_all'] = whole_df['operation'].fillna(whole_df['operation_far'])
# whole_df['commision_all'] = whole_df['commision'].fillna(whole_df['commision_far'])
# whole_df["cp"] = (whole_df['operation_all'] - whole_df['commision_all']).cumsum()
cp_idx = [i for i in range(1, len(whole_df), 2)]
real_pnl = whole_df["cp"].iloc[cp_idx].dropna()
plt.plot(real_pnl.values)

whole_df.groupby(['date']).count()

buy_df = whole_df[(whole_df['day_turn'] == 1) & (whole_df['buy'].notnull())]
sell_df = whole_df[(whole_df["day_turn"] == -1) & (whole_df['sell'] != 0)]
long_df = whole_df[(whole_df["day_turn"] == -1) & (whole_df['long'] != 0)]
short_df = whole_df[(whole_df["day_turn"] == 1) & (whole_df["short"].notnull())]
buy_n = len(buy_df)
sell_n = len(sell_df)
long_n = len(long_df)
short_n = len(short_df)

far_df = long_df.append(short_df)
far_df['id'] = np.nan
far_df['id'].loc[:long_n] = [i for i in range(0, long_n * 2, 2)]
far_df['id'].loc[long_n:] = [i for i in range(1, short_n* 2 + 1, 2)]
sort_far_df = far_df.sort_values(['id'])
sort_far_df['cp_new'] = (sort_far_df['operation_far'] - sort_far_df['commision_far']).cumsum()
far_idx = [i for i in range(1, len(far_df), 2)]
far_pnl = sort_far_df["cp_new"].iloc[far_idx]
far_pnl.plot()

buy_df = whole_df[(whole_df['day_turn'] == 1) & (whole_df['buy'].notnull())]
sell_df = whole_df[(whole_df["day_turn"] == -1) & (whole_df['sell'] != 0)]

df = buy_df.append(sell_df)
df['id'] = np.nan
df['id'].loc[:buy_n] = [i for i in range(0, buy_n * 2, 2)]
df['id'].loc[buy_n:] = [i for i in range(1, sell_n*2 + 1, 2)]
sort_df = df.sort_values(['id'])
sort_df['cp_new'] = (sort_df['operation'] - sort_df['commision']).cumsum()
idx = [i for i in range(1, len(df) + 1, 2)]
main_pnl = sort_df['cp_new'].iloc[idx]
main_pnl.plot()


real_pnl.index = pd.to_datetime(real_pnl.index)
fig, ax = plt.subplots()
ax.plot(real_pnl)
fig.autofmt_xdate()

# fig, ax = plt.subplots(figsize = (20, 12))
# ax.plot(real_pnl.values)
# ax.set_axisbelow(True)
# plt.ylabel("profit & loss", size=20)
# plt.xlabel("trading amounts", size=20)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# plt.title("20190423-20200701 rolling model, k=" + str(k), size = 25)


# pnl_df = pd.read_csv(r'...', index_col = 0)

# pnl_df['datetime'] = pnl_df['datetime.1'].apply(
#             lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
# pnl_df = pnl_df[~pnl_df['datetime'].duplicated()]
# pnl_df.index = pnl_df['datetime']
# pnl_df['diff'] = pnl_df['cp'].diff()
# mkt_pnl = pnl_df['diff']
# mkt_pnl.index = pd.to_datetime(pnl_df['datetime'])
# mkt_pnl.cumsum().plot()

# fig, ax = plt.subplots()
# ax.plot(real_pnl)
# ax.plot(mkt_pnl.cumsum())
# fig.autofmt_xdate()
# plt.title("20190423-20200701 IF (no commission fee)")
# plt.savefig(r'...')

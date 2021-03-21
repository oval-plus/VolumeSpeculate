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

def get_data(data):
    data = data.dropna(axis = 0, how = 'any')
    
    df = pd.DataFrame()
    df['interest_rate'] = data['cp'].pct_change() # interest rate
    df['intst_rate_drift'] = df['interest_rate'].shift(-1)
    return df

def get_interest_rate(main_data, far_data):
    check_df = pd.DataFrame(index = main_data.index)
    check_df['datetime'] = main_data.index
    check_df['cp'] = main_data['cp']
    check_df['cp_far'] = far_data['cp']
    check_df['interest_rate'] = check_df['cp'] / check_df['cp_far'] * (1/(250*24*3600*2))
    check_df['intst_rate_drift'] = check_df['interest_rate'].shift(-1)
    return check_df

def generate_date_range(date):
    date_range = pd.date_range(start = date + ' 09:30:00', end = date + ' 11:30:00', freq = "0.5S")
    date_range_n = pd.date_range(start = date + ' 13:00:00', end = date + ' 15:00:00', freq = "0.5S")
    date_range = date_range.append(date_range_n)
    date_range = date_range.map(lambda x: str(x))
    date_range = date_range.map(lambda x: x + ".000" if x[-7] != '.' else x[:-3])
    return date_range

def process_orig_data(date, data, t = 1):
    date_range = generate_date_range(date)
    ticker_lst = data[~data['ticker'].duplicated()]['ticker']

    main_data = data[data['ticker'] == ticker_lst.iloc[t-1]]
    main_data['datetime'] = main_data['datetime'].apply(
        lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

    main_data = main_data[~main_data['datetime'].duplicated()]
    main_data.index = main_data['datetime']
    main_data = main_data.reindex(index = date_range).fillna(method = 'ffill')
    return main_data

def calculate_param_df(df):
    df['Sx'] = df['interest_rate'].cumsum().shift()
    df['Sy'] = df['intst_rate_drift'].cumsum().shift()
    df['r2'] = df['interest_rate'].apply(lambda x: pow(x, 2))
    df['r(i-1)^2'] = df['intst_rate_drift'].apply(lambda x: pow(x, 2))
    df['Sxx'] = df['r2'].cumsum().shift()
    df['Sxy'] = (df['intst_rate_drift'] * df['interest_rate']).cumsum().shift()
    df['Syy'] = df['r(i-1)^2'].cumsum().shift()
    df['n'] = [i for i in range(0, len(df))]
    df['Sx2'] = df['Sx'].apply(lambda x: pow(x, 2))
    df['Sy2'] = df['Sy'].apply(lambda x: pow(x, 2))
    df['a_up'] = (df['n'] * df['Sxy'] - df['Sx'] * df['Sy'])
    df['a_down'] = (df['n'] * df['Sxx'] - df['Sx2'])
    df['a'] = df['a_up'] / df['a_down']
    df['b'] = (df['Sy'] - df['a'] * df['Sx']) / df['n']
    df['sd_up'] = df['n']* df['Syy'] - df['a']* (df['n'] * df['Sxy'] - df['Sx'] * df['Sy']) - df['Sy2']
    df['sd_down'] = df['n'] * (df['n'] - 2)
    df['sd'] = (df['sd_up'] / df['sd_down']).apply(np.sqrt)
    df['lambda'] = -df['a'].apply(np.log) / (1/(250 * 24 * 3600 * 2))
    df['mu'] = df['b'] / (1 - df['a'])
    df['sigma'] = df['sd'] * (-2 * df['a'].apply(np.log) / ((1/(250 * 24 * 3600 * 2)) *(
            1 - df['a'].apply(lambda x: pow(x, 2))))).apply(np.sqrt)
    df['normal'] = np.random.normal(0, 1, len(df))
    df['predict'] = df['a'] * df['interest_rate'] + df['b'] + df['sd'] * df['normal']
    return df

if __name__ == "__main__":
    config_path = r"..."
    with open(config_path, 'r') as f:
        config = json.load(f)

    util = Utility(config)
    start_date = '20190423'
    # end_date = '20200701'
    end_date = "20190601"
    date_lst = util.generate_date_lst(start_date, end_date)

    n = len(date_lst)
    whole_df = pd.DataFrame()
    for i in range(0, n):
        date = date_lst[i]
        print(date)
        data = util.open_data(date)
        if data.empty:
            continue
        data = process_orig_data(date, data, t=1)

        df = get_data(data)
        df = calculate_param_df(df)

        df['mask'] = df['predict'].shift().diff()

        # pos_idx = df[df['predict'] > df['intst_rate_drift']].index
        # df['signal'] = np.nan
        # df['signal'].loc[pos_idx] = 1
        # neg_idx = df[df['predict'] < df['intst_rate_drift']].index
        # df['signal'].loc[neg_idx] = -1
        pos_idx = df[df['mask'] < 0].index
        neg_idx = df[df['mask'] > 0].index
        df['signal'] = np.nan
        df['signal'].loc[pos_idx] = 1
        df['signal'].loc[neg_idx] = -1

        flag = -1
        for i in range(0, len(df)):
            if df['signal'].iloc[i] == 1 and flag == 1:
                df['signal'].iloc[i] = np.nan
                continue
            if df['signal'].iloc[i] == -1 and flag == -1:
                df['signal'].iloc[i] = np.nan
                continue
            if df['signal'].iloc[i] == 1 and flag == -1:
                flag = 1
                continue
            if df['signal'].iloc[i] == -1 and flag == 1:
                flag = -1
                continue

        sums = df['signal'].sum()
        if sums != 0:
            last_valid_idx = df['signal'].last_valid_index()
            if df['signal'].loc[last_valid_idx] == 1:
                df['signal'].loc[last_valid_idx] = np.nan
            elif df['signal'].loc[last_valid_idx] == -1:
                df['signal'].loc[last_valid_idx] = np.nan

        operation_df = pd.DataFrame()
        operation_df['operation'] = df['signal'] * data['cp']
        operation_df["commission"] = 0.000345 * data["cp"]
        operation_df["signal"] = df['signal']
        operation_df["datetime"] = data['datetime']
        operation_df["date"] = date

        whole_df = whole_df.append(operation_df)

    # print(df)
    pnl = (whole_df['operation'] - whole_df['commission']).dropna().cumsum()
    idx = whole_df[whole_df["signal"] == -1].index
    real_pnl = pnl.loc[idx]
    plt.plot(real_pnl.values)
    plt.plot(whole_df['operation'].cumsum().loc[idx].values)
    plt.title("vasicek model short rate no daily turn 20190423-20190601")
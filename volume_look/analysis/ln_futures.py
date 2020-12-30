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
    futures['datetime'] = pd.to_datetime(futures['datetime'])
    futures = futures.resample('1T', on = 'datetime').first()
    futures = futures.dropna(axis = 0, how = 'all')
    futures = futures[~futures.datetime.duplicated()]
    date_time = futures['datetime'].apply(lambda x: str(x)[:-4])
    futures.index = date_time.index
    futures.index = futures.index.astype(str)
    futures = futures.reindex(index = idx)
    return futures

def get_division_ratio(futures_1, futures_2):
    """ln((T-t) / (2T-t))"""
    df = pd.DataFrame()
    df['ratio'] = futures_1 / futures_2
    df['ratio'] = df['ratio'].apply(lambda x: np.log(x))
    return df

def save_ln(util, start_date, end_date):
    secs_two_week = 14 * 24 * 60 * 60
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
            ln_df[ticker + '_remain_secs'] = futures_df['remain_secs']
            ln_df[ticker + '_remain_mius'] = futures_df['remain_secs'] / 60
            # ln_df[ticker + '_remain_secs_re'] = futures_df['remain_secs'].apply(lambda x: 1/float(x))
            ln_df[ticker + '_T-t'] = futures_df['remain_secs'].apply(lambda x: secs_two_week / float(x))
        
        for j in range(0, len(ticker_lst) - 1):
            ticker_close = ticker_lst[j]
            ticker_far = ticker_lst[j + 1]
            ln_ratio = get_division_ratio(ln_df[ticker_close], ln_df[ticker_far])
            ln_df[ticker_close + '_ratio'] = ln_ratio['ratio']
        
        ln_df['ln_spot'] = get_division_ratio(ln_df['spot'], ln_df[ticker_lst[0]])

        ln_df.to_csv('/home/lky/volume_speculate/output/pair_trading/ln_futures/' + date + '.csv')

def plotly_multiply(paint_lst, name):
    trace_lst = []
    trace = go.Scatter(
            x = [i for i in range(0, len(paint_lst))],
            y = paint_lst.values,
            mode = "lines+markers",
            name = name
        )
    trace_lst.append(trace)
    layout = go.Layout(
        title = name
    )
    plotly.offline.iplot({"data": trace_lst, 
                        "layout": layout},
                        filename = 'plotly paint')
                    

if __name__ == "__main__":
    

    start_date = '20200103'
    end_date = '20200303'

    util = Utility(config)
    date_lst = util.generate_date_lst(start_date, end_date)
    # fig, ax = plt.subplots(1, 1)
    # for i in range(0, len(date_lst)):
    #     date = date_lst[i]
    #     path = '/home/lky/volume_speculate/output/pair_trading/ln_futures/{date}.csv'.format_map(vars())
    #     df = pd.read_csv(path, index_col = 0)
    #     col_lst = df.columns
    #     ticker = col_lst[2]
    #     r_df = df[ticker] * df[ticker + '_T-t']
    #     # df[ticker]: ln(futures1(T - t) / futures2(T - 2t))
    #     ticker_2 = col_lst[6]
    #     r_df_2 = df[ticker_2] * df[ticker_2 + '_T-t']
    #     plotly_multiply(r_df, ticker + " " + date)
    #     plotly_multiply(r_df_2, ticker_2 + " " + date)

        # ax.plot(r_df)
    # plt.title(start_date + ' - ' + end_date)
    save_ln(util, start_date, end_date)

    df = pd.read_csv('/home/lky/volume_speculate/output/pair_trading/ln_futures/20200107.csv', index_col = 0)
    # r = df['ln_spot']
    # r2 = df['IF2001_ratio']
    r = df['IF2003_ratio'] / (df['IF2003_remain_secs'])
    r2 = df['IF2002_ratio'] / (df['IF2002_remain_secs'])

    r.index = pd.to_datetime(r.index, format = '%Y-%m-%d %H:%M:%S')
    r2.index = pd.to_datetime(r2.index, format = '%Y-%m-%d %H:%M:%S')
    # r3.index = pd.to_datetime(r2.index, format = '%Y-%m-%d %H:%M:%S')

    fig, ax = plt.subplots(1, 1)
    ax.plot(r, label = 'tao_1 tao_2')
    ax.plot(r2, label = 'tao2, tao3')
    plt.legend(loc = 'best')
    fig.autofmt_xdate()

    # r_df = pd.DataFrame()
    # r2_df = pd.DataFrame()
    # path = '/home/lky/volume_speculate/output/pair_trading/ln_futures/'
    # fsinfo = sorted(os.listdir(path))
    # for f in fsinfo:
    #     df_r = pd.DataFrame()
    #     df_r2 = pd.DataFrame()
    #     temp_path = path + f
    #     df = pd.read_csv(temp_path, index_col = 0)
    #     df_r['ln_spot'] = df['ln_spot']
    #     df_r2['ratio'] = df.iloc[:, -4]

    #     r_df = r_df.append(df_r)
    #     r2_df = r2_df.append(df_r2)

    # # fig, ax = plt.subplots(1, 1)
    # # ax.plot(r_df['ln_spot'].values, label = 'spot tao_1')
    # # ax.plot(r2_df['ratio'].values, label = 'tao1, tao2')
    # # plt.legend(loc = 'best')

    # trace_spot = go.Scatter(
    #         x = [i for i in range(0, len(r_df))],
    #         y = r_df['ln_spot'].values,
    #         mode = "lines+markers",
    #         name = "spot, tao_1"
    #     )
    # # trace_tao = go.Scatter(
    # #     x = [i for i in range(0, len(r2_df))],
    # #     y = r2_df['ratio'].values,
    # #     mode = "lines+markers",
    # #     name = "tao_1, tao_2"
    # #     )
    # trace_lst = [trace_spot]
    # layout = go.Layout(
    #     title = "title"
    # )
    # plotly.offline.iplot({"data": trace_lst, 
    #                     "layout": layout},
    #                     filename = 'plotly paint')


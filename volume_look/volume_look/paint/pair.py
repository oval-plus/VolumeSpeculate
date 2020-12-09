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
# diff = pd.read_csv('/home/lky/volume_speculate/output/pair_trading/ratio/together/20190603_20191231_0.csv', index_col = 0)
# diff = diff.set_index(pd.to_datetime(diff['date'], format = '%Y%m%d'))
# sign = diff[~diff['ticker'].duplicated()]
# sign = sign['ratio']

# fig, ax = plt.subplots(1, 1)
# ax.plot(diff['ratio'])
# ax.scatter(sign.index, sign, color = 'red')
# fig.autofmt_xdate()

class PairPaint(object):
    def __init__(self, start_date, end_date, config, paint):
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.paint = paint

    # @staticmethod
    def generate_whole_df(self, date_lst, k):
        whole_df = pd.DataFrame()
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            path = "/home/lky/volume_speculate/output/pair_trading/spot_futures/{k}/{date}.csv".format_map(vars())
            df = pd.read_csv(path)
            whole_df = whole_df.append(df)
        t = int(k) + 1
        save_path = "/home/lky/volume_speculate/output/pair_trading/spot_futures/whole/whole_{t}_{self.start_date}_{self.end_date}.csv".format_map(vars())
        whole_df.to_csv(save_path)

    def read_whole_df(self, k):
        path = self.whole_path(k)
        whole_df = pd.read_csv(path)
        whole_df['date'] = whole_df['datetime'].apply(lambda x: x[:10])
        whole_df['datetime'] = pd.to_datetime(whole_df['datetime'], format = '%Y-%m-%d %H:%M:%S')
        whole_df = whole_df.set_index(['datetime'])
        whole_df = whole_df.dropna(axis = 0, how = 'any')
        whole_df = whole_df[~whole_df.index.duplicated()]
        return whole_df

    # @staticmethod
    def whole_path(self, k):
        path = '/home/lky/volume_speculate/output/pair_trading/spot_futures/whole/whole_{k}_{self.start_date}_{self.end_date}.csv'.format_map(vars())
        return path

    def get_basis(self, whole_df):
        gap = whole_df['stock'] - whole_df['futures']
        return gap

    def get_futures_gap(self, futures_1, futures_2):
        gap = futures_1['futures'] - futures_2['futures']
        return gap

    def get_date_series(self, whole_df):
        new_date = []
        whole_df_new = whole_df.reset_index()
        date_drop_dup = whole_df_new[~whole_df_new['date'].duplicated(keep = 'first')]
        last_drop_dup = whole_df_new[~whole_df_new['date'].duplicated(keep = 'last')]
        sign_gap = whole_df[~whole_df['ticker'].duplicated(keep = 'last')]
        sign_df = pd.DataFrame()
        sign_df['datetime'] = sign_gap.index
        sign_df['date'] = sign_df['datetime'].apply(lambda x: str(x)[:10])
        for i in range(0, len(sign_df)):
            date = sign_df['date'].iloc[i]
            idx = date_drop_dup[date_drop_dup['date'] == date].index[0]
            idx_ = last_drop_dup[last_drop_dup['date'] == date].index[0]
            if idx == 0 or idx_ == 0:
                continue
            last_date = last_drop_dup['date'].shift(10).loc[idx_]
            last_idx = last_drop_dup[last_drop_dup['date'] == last_date].index[0]
            new_date += [j for j in range(last_idx, idx_ + 1)]
        return new_date

    def last_two_weeks(self, whole_df):
        date_new = self.get_date_series(whole_df)
        whole_df_new = whole_df.reset_index()
        whole_df_new = whole_df_new.reindex(index = date_new)
        whole_df_new = whole_df_new.set_index(['datetime'])
        whole_df_new = whole_df.loc[whole_df_new.index]
        return whole_df_new

    def get_sign_gap(self, gap, whole_df):
        sign_gap = whole_df[~whole_df['ticker'].duplicated(keep = 'last')]
        sign_gap = gap.loc[sign_gap.index]
        return sign_gap

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(gap)
    # ax.scatter(sign_gap_1.index, sign_gap_1, color = 'red')
    # fig.autofmt_xdate()
    # plt.title('spot and closest futures basis situation')

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(gap_futures, label = 'tao_1, tao_2 gap')
    # ax.plot(gap, label = 'spot, tao_1 gap')
    # plt.vlines(sign_gap_1.index, -50, 50, colors = 'green', linestyles = 'dashed')
    # fig.autofmt_xdate()
    # plt.legend(loc = 'best')
    # plt.title('tao_1, tao_2 gap in whole time series')

    def paint_part_gap(self, futures_12, gap, ticker):
        fig, ax = plt.subplots(1, 1)
        ax.plot([i for i in range(0, len(futures_12))], futures_12, label = 'tao_1, tao_2 gap')
        ax.plot([i for i in range(0, len(gap))], gap, label = 'spot, tao_1 gap')
        plt.legend(loc = 'best')
        plt.title('gap tao_1 = ' + ticker)
        save_path = '/home/lky/volume_speculate/output/pair_trading/pic/gap/'
        plt.savefig(save_path + str(ticker) + '.png')
    
    def plotly_part_gap(self, futures_12, gap, ticker):
        fig = go.Figure()
        trace_futures = go.Scatter(
            x = [i for i in range(0, len(futures_12))],
            y = futures_12,
            mode = "lines+markers",
            name = 'tao_1, tao_2 gap'
        )
        trace_gap = go.Scatter(
            x = [i for i in range(0, len(gap))],
            y = gap,
            mode = "lines+markers",
            name = 'spot, tao_1 gap'
        )
        trace_lst = [trace_futures, trace_gap]
        layout = go.Layout(
            title = ticker            
        )
        save_path = "/home/lky/volume_speculate/output/pair_trading/pic/gap/plotly/"
        plotly.offline.iplot({"data": trace_lst, 
                            "layout": layout},
                            filename = save_path + ticker + '.png',
                            image = 'png')


    def matplotlib_part(self, futures_2, futures_1, spot, ticker):
        fig, ax = plt.subplots(1, 1)
        ax.plot([i for i in range(0, len(futures_1))], futures_1, label = 'tao_1')
        ax.plot([i for i in range(0, len(futures_2))], futures_2, label = 'tao_2')
        ax.plot([i for i in range(0, len(spot))], spot, label = 'spot')
        plt.legend(loc = 'best')
        plt.title('tap_1 = ' + ticker)
    
    def plotly_part(self, futures_2, futures_1, spot, ticker):
        """paint the last two weeks of the futures and spot price 
        in each picture with plotly
        """
        # fig = go.Figure()
        trace_futures1 = go.Scatter(
            x = [i for i in range(0, len(futures_1))],
            y = futures_1,
            mode = "lines+markers",
            name = 'tao_1'
        )
        trace_futures2 = go.Scatter(
            x = [i for i in range(0, len(futures_2))],
            y = futures_2,
            mode = "lines+markers",
            name = 'tao_2'
        )
        trace_spot = go.Scatter(
            x = [i for i in range(0, len(spot))],
            y = spot,
            mode = "lines+markers",
            name = 'spot'
        )
        trace_lst = [trace_futures1, trace_futures2, trace_spot]
        layout = go.Layout(
            title = ticker            
        )
        # save_path = "/home/lky/volume_speculate/output/pair_trading/pic/gap/plotly/
        plotly.offline.iplot({"data": trace_lst, 
                            "layout": layout},
                            filename = ticker)


    def gather_last_week(self, last_whole_df_1, last_whole_df_2):
        ticker_lst = last_whole_df_1['ticker'].unique()
        n = len(ticker_lst) - 2
        fig, ax = plt.subplots(n, 1, figsize = (8, 12))
        for t in range(0, n):
            ticker = ticker_lst[t]
            part_1 = last_whole_df_1[last_whole_df_1['ticker'] == ticker]
            date_lst = part_1['date'].unique()
            part_2 = last_whole_df_2[last_whole_df_2['date'].isin(date_lst)]
            gap_part = self.get_basis(part_1)
            futures_12_part = self.get_futures_gap(part_1, part_2)

            ax[t].plot([i for i in range(0, len(futures_12_part))], futures_12_part, label = 'tao_1, tao_2 gap')
            ax[t].plot([i for i in range(0, len(gap_part))], gap_part, label = 'spot, tao_1 gap')
            ax[t].set_title('gap tao_1 = ' + ticker)
        plt.subplots_adjust(wspace = 0, hspace = 0.5)
        plt.legend(loc = 'best')

    def paint_last_week(self, last_whole_df_1, last_whole_df_2, func):
        """use plotly or matplotlib to paint the last two weeks' futures gap and spot market gap
            in each image"""
        ticker_lst = last_whole_df_1['ticker'].unique()
        for ticker in ticker_lst[:-2]:
            part_1 = last_whole_df_1[last_whole_df_1['ticker'] == ticker]
            date_lst = part_1['date'].unique()
            part_2 = last_whole_df_2[last_whole_df_2['date'].isin(date_lst)]
            gap_part = self.get_basis(part_1)
            futures_12_part = self.get_futures_gap(part_1, part_2)
            func(futures_12_part, gap_part, ticker)
            # paint_part(part_2['futures'], part_1['futures'], part_1['stock'], ticker)
        
    def whole_last_week(self, last_gap_futures, last_gap, last_sign_gap):
        fig, ax = plt.subplots(1, 1)
        ax.plot(last_gap_futures, label = 'tao_1, tao_2 gap')
        ax.plot(last_gap, label = 'spot, tao_1 gap')
        plt.vlines(last_sign_gap.index, -50, 50, colors = 'green', linestyles = 'dashed')
        # fig.autofmt_xdate()
        plt.legend(loc = 'best')
        plt.title("last 2 weeks tao_1, tao_2 gap")

    def trigger(self):
        util = Utility(self.config)
        date_lst = util.generate_date_lst(self.start_date, self.end_date)

        # self.generate_whole_df(date_lst, 0)
        # self.generate_whole_df(date_lst, 1)
        whole_df_1 = self.read_whole_df(1)
        whole_df_2 = self.read_whole_df(2)
        # whole_df_3 = self.read_whole_df(3)
        gap = self.get_basis(whole_df_1)
        gap_futures = self.get_futures_gap(whole_df_1, whole_df_2)

        last_whole_df_2 = self.last_two_weeks(whole_df_2)
        last_whole_df_1 = self.last_two_weeks(whole_df_1)
        # last_whole_df_3 = self.last_two_weeks(whole_df_3)
        last_gap_futures = self.get_futures_gap(last_whole_df_1, last_whole_df_2)
        last_gap = self.get_basis(last_whole_df_1)
        last_sign_gap = self.get_sign_gap(last_gap, last_whole_df_1)

        if self.paint == 'matplotlib':
            func = self.paint_part_gap
        elif self.paint == "plotly":
            func = self.plotly_part_gap
        self.paint_last_week(last_whole_df_1, last_whole_df_2, func)
        self.whole_last_week(last_gap_futures, last_gap, last_sign_gap)

if __name__ == "__main__":
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    start_date = '20190601'
    end_date = '20200601'
    paint = 'matplotlib'

    pp = PairPaint(start_date, end_date, config, paint)
    pp.trigger()



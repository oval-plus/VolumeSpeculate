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

class PaintCal(object):
    def __init__(self, start_date, end_date, k, config, cal_type):
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.k = k
        self.cal_type = cal_type

    @staticmethod
    def cal_dts(df):
        dts = (df['ts'] - df['ts'].shift())
        return dts.dropna()

    @staticmethod
    def cal_dp(df):
        dp = (df['cp'].shift(-1) - df['cp']) / df['cp']
        return dp.dropna()

    def gather_data(self, util, date_lst, k, func):
        whole_df = pd.DataFrame()
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            data = BuildData().cal_remain_maturity(util, date)
            ticker_lst = data['ticker'].unique()
            ticker = ticker_lst[k]
            df = data[data.ticker == ticker]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[df['sod'] < 40800]
            df = df.resample('1T', on='datetime').first()

            whole_df[date] = func(df).values
        return whole_df

    def matplotlib_paint(self, df):
        fig, ax = plt.subplots(1, 1)
        for col in df.columns:
            paint_lst = df[col]
            ax.plot([i for i in range(len(paint_lst))], paint_lst.values, label=col)
        plt.legend(loc='best')

    def plotly_paint(self, df):
        trace_lst = []
        for col in df.columns:
            paint_lst = df[col]
            trace = go.Scatter(
                x = [i for i in range(0, len(paint_lst))],
                y = paint_lst.values,
                mode = "lines+markers",
                name = col
            )
            trace_lst.append(trace)        
        layout = go.Layout(
            title = 'morning {self.start_date} - {self.end_date} {self.cal_type}'.format_map(vars())
        )
        plotly.offline.iplot({"data": trace_lst, 
                            "layout": layout},
                            filename = 'plotly paint')
    

    def trigger(self):
        if self.cal_type == 'dts':
            func = self.cal_dts
        elif self.cal_type == 'dp':
            func = self.cal_dp
        util = Utility(self.config)
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        df = self.gather_data(util, date_lst, k, func)
        # self.plotly_paint(df)
        # self.matplotlib_paint(util, date_lst, k)

if __name__ == "__main__":
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    start_date = '20200103'
    end_date = '20200103'

    k = 0
    cal_type = 'dts'
    pc = PaintCal(start_date, end_date, k, config, cal_type)
    pc.trigger()

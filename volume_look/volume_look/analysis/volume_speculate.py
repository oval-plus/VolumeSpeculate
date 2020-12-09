# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from volume_look.util.utility import Utility
from volume_look.analysis.build_data import BuildData

class VolumeSpeculate(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def total_volume(self, util, date):
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        if data is None:
            return [0], [0], pd.Series([0])
        ticker_lst = data['ticker'].unique()
        ts_lst = []
        remain_date_lst = []
        for ticker in ticker_lst:
            temp_data = data[data['ticker'] == ticker]
            ts = temp_data['ts'].iloc[-1]
            remain_date = temp_data['remain_date'].iloc[-1]
            remain_date_lst.append(remain_date)
            ts_lst.append(ts)
        return remain_date_lst, ts_lst, ticker_lst

    def iteration_volume(self, util, start_date, end_date):
        date_lst = util.generate_date_lst(start_date, end_date)
        remain_date_lst, ts_lst, ticker_lst = [], [], []
        dt_lst = []
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            remain_date, ts, ticker = self.total_volume(util, date)
            remain_date_lst += remain_date
            ts_lst += ts
            ticker_lst += ticker.tolist()
            dt_lst += [date] * len(ts)
        df = pd.DataFrame()
        df['ts'] = ts_lst
        df['remain_date'] = remain_date_lst
        df['date'] = dt_lst
        df['ticker'] = ticker_lst
        return df

    def arma(self, util):
        pass
    
    def cal_corr(self, util, date, k):
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        data['datetime'] = data['datetime'].apply(lambda x: x[:-6] + '00')
        data = data.set_index(['datetime'])
        data = data[~data.index.duplicated(keep='first')]
        data.index = data['sod']
        if k == 0:
            back_date = date
        else:
            back_date = util.get_week_date(date, k)
        stock_price = util.get_stock_price(back_date)
        idx = stock_price['sod']
        data = data.reindex(index = idx)
        stock_price = stock_price.set_index(['sod'])
        corr_df = pd.DataFrame()
        corr_df['cp'] = data['cp']
        corr_df['stock'] = stock_price['close']
        corr = corr_df.corr()
        return corr['stock'].loc['cp']
    
    def get_big_data(self, util):
        bd = BuildData()
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        whole_data = pd.DataFrame()
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            df = bd.cal_remain_maturity(util, date)
            whole_data = whole_data.append(df)
        return whole_data

    def mature_corr(self, util, date, k):
        # bigdata = self.get_big_data(util)
        # data = bigdata[(bigdata['remain_date'] <= k + 1) & (bigdata['remain_date'] >= k - 1)]
        bd = BuildData()
        stock_price = util.get_stock_price(date)
        data = bd.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        corr_df = pd.DataFrame()

        for ticker in ticker_lst:
            tmp_data = data[data['ticker'] == ticker]
            tmp_data.index = tmp_data['sod']
            tmp_data = tmp_data[~tmp_data.index.duplicated(keep='first')]
            corr_df[ticker] = tmp_data['cp']

        stock_price.index = stock_price['sod']
        corr_df['stock'] = stock_price['close']
        stock = corr_df['stock'].dropna()
        corr_df = corr_df.reindex(index = stock.index).fillna(method = 'ffill')
        corr = corr_df.corr()
        corr['remain_date'] = data['remain_date'].unique().tolist() + [0]
        return corr        
    
def main():
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    start_date = '20200101'
    end_date = '20200701'
    util = Utility(config)
    vs = VolumeSpeculate(start_date, end_date)
    date_lst = util.generate_date_lst(start_date, end_date)
    corr_lst = []
    k = 0
    for i in range(0, len(date_lst)):
        date = date_lst[i]
        # corr = vs.cal_corr(util, date, k)
        corr = vs.mature_corr(util, date, k)
        corr_lst.append(corr)
    df = pd.DataFrame()
    df['corr'] = corr_lst
    path = os.path.join(config['prefix'], config['dir_path']['corr'], str(k) + '_mature.csv')
    df.to_csv(path)
    return corr_lst


if __name__ == "__main__":
    main()

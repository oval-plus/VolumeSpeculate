# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.api import ARIMA
from volume_look.util.utility import Utility

config_path = r'D:\Document\SJTU\thesis\program\volume_look\config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

class VolumeSpeculate(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def get_main_contract(self, util, date):
        """get the main contract"""
        data = util.open_data(date)
        if data.empty:
            return None
        start_time = util.shen_start_time(date, True)
        temp = data[data.sod < (start_time - 60)]

        if temp['oi'].empty:
            temp = data[data.sod == start_time]
            temp = temp[temp.ts == max(temp.ts)]
        else:
            temp = temp[temp.oi == max(temp.oi)] 
        return temp['ticker'].iloc[0]

    def get_main_data(self, util, date):
        """get the data frame"""
        data = util.open_data(date)
        instrument = self.get_main_contract(util, date)

        start_time = util.get_start_time(date)
        end_time = util.get_end_time(date)
        
        idx = data[(data['ticker'] == instrument) & (data['sod'] >= start_time) & (
                        data['sod'] < end_time)].index.tolist()
        main_data = data.loc[idx, :]
        main_data = main_data.dropna()
        return main_data

    def cal_remain_maturity(self, util, date):
        remain_dict = dict()
        data = util.open_data(date)
        if data.empty:
            return None
        ticker_lst = data['ticker'].unique()
        mature_dict = dict()
        for ticker in ticker_lst:
            maturity_date = util.get_ticker_maturity(ticker)
            maturity_sod = maturity_date + ' ' + util.last_trade_time()
            remain = (datetime.datetime.strptime(maturity_date, '%Y%m%d') - datetime.datetime.strptime(date, '%Y%m%d')).days
            remain_dict[ticker] = remain
            mature_dict[ticker] = maturity_sod

        data['remain_date'] = data['ticker']
        data = data.replace({'remain_date': remain_dict})
        data['mature'] = data['ticker']
        data = data.replace({"mature": mature_dict})

        datetime_secs = data['datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        datetime_mature = data['mature'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S'))

        data['remain_secs'] = datetime_mature - datetime_secs
        data['remain_secs'] = data['remain_secs'].apply(lambda x: x.days * 24 * 3600 + x.seconds)
        return data

    def total_volume(self, util, date):
        data = self.cal_remain_maturity(util, date)
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
    
    def cal_corr(self, util, date):
        data = self.cal_remain_maturity(util, date)
        data['datetime'] = data['datetime'].apply(lambda x: x[:-4])
        data = data.set_index(['datetime'])
        stock_price = util.get_stock_price()
        new_date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        date_stock_price = stock_price[stock_price.index.str.contains(new_date)]
        corr = np.corrcoef(date_stock_price, data)
        return corr
    
def main():
    config_path = r"D:\Document\SJTU\thesis\program\volume_look\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    start_date = '20200101'
    end_date = '20200701'
    util = Utility(config)
    vs = VolumeSpeculate(start_date, end_date)
    date_lst = util.generate_date_lst(start_date, end_date)
    corr_lst = []
    for i in range(0, date_lst):
        date = date_lst[i]
        corr = vs.cal_corr(util, date)
        corr_lst.append(corr)
    return corr_lst

if __name__ == "__main__":
    main()



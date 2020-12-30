# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from cachetools import TTLCache, cached
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from volume_look.util.utility import Utility

class BuildData(object):
    def __init__(self):
        self.t = 90

    @cached(cache=TTLCache(maxsize = 100, ttl = 1200))
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
        
        data['updatetime'] = data['datetime'].apply(lambda x: str(x)[11:])

        data['remain_date'] = data['ticker']
        data = data.replace({'remain_date': remain_dict})
        data['mature'] = data['ticker']
        data = data.replace({"mature": mature_dict})

        datetime_secs = data['datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        datetime_mature = data['mature'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S'))

        data['remain_secs'] = datetime_mature - datetime_secs
        data['remain_secs'] = data['remain_secs'].apply(lambda x: x.days * 24 * 3600 + x.seconds)
        return data

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

    def get_mid_price(self, main_data):
        mid_price = (main_data['b1'] + main_data['s1']) / 2
        return mid_price
    
    def build_main_data(self, util, date, ticker):
        data = util.open_data(date)
        start_time = util.get_start_time(date)
        end_time = util.get_end_time(date)

        idx = data[(data['ticker'] == ticker) & (data['sod'] >= start_time) & (
                        data['sod'] < end_time)].index.tolist()
        main_data = data.loc[idx, :]
        main_data = main_data.dropna()
        return main_data
    
    def get_ask(self, main_data, mid_price, trade_at_mid):
        """get ask price"""
        ask_df = pd.DataFrame()
        if trade_at_mid:
            ask_df["ask"] = mid_price
        else:
            ask_df["ask"] = main_data["s1"]
        ask_df["sod"] = main_data["sod"]
        return ask_df
    
    def get_bid(self, main_data, mid_price, trade_at_mid):
        """get bid price"""
        bid_df = pd.DataFrame()
        if trade_at_mid:
            bid_df["bid"] = mid_price
        else:
            bid_df["bid"] = main_data["b1"]
        bid_df["sod"] = main_data["sod"]
        return bid_df
    
    def _get_time_secs(self, main_data):
        time_secs = main_data["sod"]  # FIXME:
        return time_secs

    def get_ind_open(self, util, main_data, morning):
        """get the index of the open time"""
        date = main_data["datetime"].iloc[0]
        date = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        open_time = util.get_open_time(date, morning)
        time_secs = self._get_time_secs(main_data).reset_index(drop=True)
        ind_open = time_secs[time_secs >= open_time].index[0]
        idx = main_data["datetime"].iloc[ind_open]
        return idx

    def get_ind_close(self, util, main_data, morning):
        """get the index of the close time"""
        date = main_data["datetime"].iloc[0]
        date = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        close_time = util.get_close_time(date, morning)
        time_secs = self._get_time_secs(main_data).reset_index(drop=True)
        if time_secs[time_secs >= close_time].empty:
            ind_close = time_secs.index[-1]
        else:
            ind_close = time_secs[time_secs >= close_time].index[0]
        # ind_close = ind_close
        idx = main_data["datetime"].iloc[ind_close]
        return idx
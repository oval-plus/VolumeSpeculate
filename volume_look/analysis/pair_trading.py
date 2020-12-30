# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from cachetools import TTLCache, cached
from volume_look.util.utility import Utility
from volume_look.util.save_util import SaveUtil
from volume_look.analysis.build_data import BuildData


class PairTrade(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def pair_trading(self, util, date, k = 0):
        """calculate the pair trading linear model"""
        if k > 2:
            # raise Exception("k is too big to have the result")
            k = 0
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        main_contract = ticker_lst[k]
        second_contract = ticker_lst[k + 1]

        main_data = data[data['ticker'] == main_contract]
        second_data = data[data['ticker'] == second_contract]
        main_mid = bd.get_mid_price(main_data)
        second_mid = bd.get_mid_price(second_data)

        main_mid.index = main_data['updatetime']
        second_mid.index = second_data['updatetime']
        second_mid = second_mid[~second_mid.index.duplicated()]

        reg_df = pd.DataFrame()
        reg_df['close'] = main_mid
        reg_df['far'] = second_mid

        reg_df = reg_df.dropna(axis = 0, how = 'any')

        model = sm.OLS(reg_df['close'], reg_df['far']).fit()
        return model
    
    def get_remain_date(self, util, date):
        data = BuildData().cal_remain_maturity(util, date)
        remain_date = data['remain_date'].unique()
        return remain_date
    
    @cached(TTLCache(maxsize = 100, ttl = 1200))
    def get_main_ticker(self, util, date, k = 0):
        data = BuildData().cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        ticker = ticker_lst[k]
        return ticker
    
    def pair_spread(self, util, date, k):
        if k > 2:
            # raise Exception("k is too big to have the result")
            k = 0
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        main_contract = ticker_lst[k]
        second_contract = ticker_lst[k + 1]

        main_data = data[data['ticker'] == main_contract]
        second_data = data[data['ticker'] == second_contract]

        main_data.index = main_data['updatetime']
        second_data.index = second_data['updatetime']
        second_data = second_data[~second_data.index.duplicated()]

        spread = main_data['cp'] - second_data['cp']
        return spread
    
    def mature_futures(self, util, date, k):
        """get the mature futures"""
        bd = BuildData()
        ticker = self.get_main_ticker(util, date, k)
        mature_date = util.get_ticker_maturity(ticker)
        
        spread = self.spot_futures(util, mature_date)
        return spread
    
    def mean_spot(self, util, date, k):
        stock_price = util.get_stock_price(date)
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        main_contract = ticker_lst[k]

        main_data = data[data['ticker'] == main_contract]
        idx = main_data['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").strftime(
                "%Y-%m-%d %H:%M:%S"))
        idx = idx.apply(lambda x: datetime.datetime.strptime(
                str(x)[:-3], "%Y-%m-%d %H:%M"
            ).strftime("%Y-%m-%d %H:%M:%S")
        )
        cp_df = main_data['cp']
        cp_df.index = idx
        stock_price = stock_price.reindex(index = idx).fillna(method = 'ffill')

        df = pd.DataFrame()
        df['futures'] = cp_df
        df['stock'] = stock_price['close']
        return df
    
    def spot_futures(self, util, date):
        stock_price = util.get_stock_price(date)
        bd = BuildData()
        data = bd.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        main_contract = ticker_lst[0]

        main_data = data[data['ticker'] == main_contract]
        idx = main_data['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").strftime(
                "%Y-%m-%d %H:%M:%S"))
        idx = idx.apply(lambda x: datetime.datetime.strptime(
                str(x)[:-3], "%Y-%m-%d %H:%M"
            ).strftime("%Y-%m-%d %H:%M:%S")
        )
        cp_df = main_data['cp']
        cp_df.index = idx
        stock_price = stock_price.reindex(index = idx).fillna(method = 'ffill')
        gap = main_data['cp'] - stock_price['close']
        gap = gap[~gap.index.duplicated()]
        return gap
    
    def mean_reversion(self, util, su, k):
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            mean_spot = self.mean_spot(util, date, k)
            remain = self.get_remain_date(util, date)[k]
            ticker = self.get_main_ticker(util, date, k)
            su.save_spot_futures(mean_spot, k, date, remain, ticker)

    def operate_futures_gap(self, util, su, k = 0, sigma = 0.0005):
        ratio_lst = []
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        ticker_lst = []
        remain_far, remain_close = [], []

        for i in range(0, len(date_lst)):
            date = date_lst[i]
            spot_futures = self.mature_futures(util, date, k = 0)
            mean = spot_futures.mean()
            spread = self.pair_spread(util, date, k = k)
            times = len(spread[abs(spread) > mean])
            ratio = times / len(spread)
            ratio_lst.append(ratio)
            
            remain = self.get_remain_date(util, date)
            remain_far.append(remain[k + 1])
            remain_close.append(remain[k])

            ticker = self.get_main_ticker(util, date, k)
            ticker_lst.append(ticker)
            # print(ratio)
        su.save_ratio(ratio_lst, k, date_lst, remain_far, remain_close, ticker_lst)

    def operate_trade(self, util, su, k):
        coef_lst = []
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        remain_far, remain_close = [], []
        for i in range(0, len(date_lst)):
            date = date_lst[i]
            # print(date)
            model = self.pair_trading(util, date, k)
            coef = model.params['far']
            resid_df = model.resid
            coef_lst.append(coef)
            remain = self.get_remain_date(util, date)
            remain_far.append(remain[k + 1])
            remain_close.append(remain[k])

            su.save_resid(resid_df, k, date)
        su.save_coef(coef_lst, k, date_lst, remain_far, remain_close)

if __name__ == "__main__":
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # start_date = '20200318'
    start_date = '20200101'
    end_date = '20200801'
    
    pt = PairTrade(start_date, end_date)
    util = Utility(config)
    su = SaveUtil(config)
    for k in range(0, 3):
    #     pt.operate_trade(util, su, k)
        # pt.operate_futures_gap(util, su, k)
        pt.mean_reversion(util, su, k)
        print(k, 'complete')
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import json

import statsmodels.api as sm
from cachetools import cached, TTLCache
from volume_look.util.utility import Utility
from volume_look.analysis.maluyao import Maluyao


class ConstructSignal(object):
    def __init__(self, morning):
        self.morning = morning
    
    def build_main_data(self, util, date, ticker):
        data = util.open_data(date)
        start_time = util.get_open_time(date, self.morning)
        end_time = util.get_close_time(date, self.morning)
        data['datetime'] = data['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')

        idx = data[(data['ticker'] == ticker) & (data['sod'] >= start_time) & (
                        data['sod'] < end_time)].index.tolist()
        main_data = data.loc[idx, :]
        main_data = main_data.dropna()
        return main_data
    
    def get_main_ticker(self, util, date, k = 0):
        """get the main ticker"""
        data = self.cal_remain_maturity(util, date)
        ticker_lst = data['ticker'].unique()
        ticker = ticker_lst[k]
        return ticker
    
    @cached(cache=TTLCache(maxsize = 100, ttl = 1200))
    def cal_remain_maturity(self, util, date):
        """calculate the remaining mature date"""
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
        
        data['updatetime'] = data['datetime'].apply(lambda x: x[11:])

        data['remain_date'] = data['ticker']
        data = data.replace({'remain_date': remain_dict})
        data['mature'] = data['ticker']
        data = data.replace({"mature": mature_dict})

        datetime_secs = data['datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        datetime_mature = data['mature'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S'))

        data['remain_secs'] = datetime_mature - datetime_secs
        data['remain_secs'] = data['remain_secs'].apply(lambda x: x.days * 24 * 3600 + x.seconds)
        return data
    
    def get_mid_price(self, main_data):
        mid_price = (main_data['b1'] + main_data['s1']) / 2
        return mid_price

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
        time_secs = main_data["sod"]
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
    
    def get_effective_price(self, main_data):
        eff_price = (main_data['s1'] * main_data['sv1'] + 
                    main_data['b1'] * main_data['bv1']) / (
                    main_data['bv1'] + main_data['sv1'])
        return eff_price
    
    def eff_signal(self, main_data):
        """effective price minus the mid-price"""
        eff_price = self.get_effective_price(main_data)
        mid_price = self.get_mid_price(main_data)
        res_df = pd.DataFrame()
        res_df['res'] = eff_price - mid_price
        res_df['datetime'] = main_data['datetime']
        res_df['res'] = res_df['res'].apply(lambda x: False if x > 0.5 else x)
        res_df['res'] = res_df['res'].apply(lambda x: True if x < -0.5 else x)
        return res_df
    
    def get_spread(self, main_data):
        spread = main_data['s1'] - main_data['b1']
        return spread
    
    def deriv_signal(self, main_data, delay):
        """effective price minus the mid-price divide the mid price"""
        eff_price = self.get_effective_price(main_data)
        mid_price = self.get_mid_price(main_data)
        res_df = pd.DataFrame()
        res_df['res'] = (eff_price.rolling(delay).mean() - mid_price.shift(delay))
        res_df['datetime'] = main_data['datetime']
        res_df['res'] = res_df['res'].apply(lambda x: True if x < -0.5 else x)
        res_df['res'] = res_df['res'].apply(lambda x: False if x > 0.5 else x)
        return res_df
    
    def div_deriv(self, main_data, delay):
        eff_price = self.get_effective_price(main_data)
        mid_price = self.get_mid_price(main_data)
        spread = self.get_spread(main_data)
        res_df = pd.DataFrame()
        res_df['res'] = (eff_price.rolling(delay).mean() - mid_price.shift(delay)) / spread
        res_df['datetime'] = main_data['datetime']
        res_df['res'] = res_df['res'].apply(lambda x: False if x < -10 else x)
        res_df['res'] = res_df['res'].apply(lambda x: True if x > 10 else x)
        return res_df

    
    def dp_signal(self, main_data):
        """derivative price"""
        dp = (main_data['cp'] - main_data['cp'].shift()) / main_data['cp'].shift()
        res_df = pd.DataFrame()
        res_df['datetime'] = main_data['datetime']
        res_df['res'] = dp
        res_df['res'] = res_df['res'].apply(lambda x: True if x < -0.000025 else x)
        res_df['res'] = res_df['res'].apply(lambda x: False if x > 0.000025 else x)
        return res_df

    def hk_signal(self, util, main_data):
        """hugangtong"""
        date = main_data['datetime'].iloc[0][:10]
        date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        
        main_data['datetime'] = main_data['datetime'].apply(
            lambda x: x[:-3] + '000' if x[-3] < '5' else x[:-3] + '500'
            )
        
        path = util.read_hk_turnover(date)
        res = util.check_hk_files(date)
        while not res:
            date = util.get_back_date(date)
            path = util.read_hk_turnover(date)
            res = util.check_hk_files(date)
        df = pd.read_csv(path, sep="\t")
        if df.empty:
            return pd.DataFrame(columns = ['res', 'datetime'])
        df_odd = df[df.index % 2 == 1]['dataTime'].apply(lambda x: x + '.000')
        df_even = df[df.index % 2 == 0]['dataTime'].apply(lambda x: x + '.500')

        datatime = df_odd.append(df_even).sort_index()
        df['datetime'] = datatime
        df['dataDate'] = df['dataDate'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
        df['datetime'] = df['dataDate'] + ' ' + df['datetime']
        new_df = df[df.direction == 'SB']
        df_sh = new_df[new_df.market == 'SH']
        df_sz = new_df[new_df.market == 'SZ']
        
        df_sh = df_sh.set_index(['datetime'])
        df_sz = df_sz.set_index(['datetime'])
        net = df_sz['net'] + df_sh['net']
        net = net.combine_first(df_sz['net'])
        net = net.combine_first(df_sh['net'])

        res_df = pd.DataFrame()
        res_df['datetime'] = net.index
        res_df['res'] = net.values
        res_df = res_df.set_index(['datetime'], drop = False)
        res_df = res_df.reindex(index = main_data['datetime'])
        res_df.index = main_data.index
        res_df = res_df.dropna(axis = 0)
        res_df['res'] = res_df['res'].apply(lambda x: True if x < 0 else False)
        return res_df
    
    def get_maluyao(self, util, main_data, date, k):
        mly = Maluyao(self.morning)
        t = util.get_maluyao_k()
        # main_data = mly.get_data(util, date, k)
        mu_df = mly.get_mut(util, main_data)
        std_df = mly.get_std(util, main_data)
        spread = mly.cal_spread(util, date, k)
        spread = spread.reindex(index = mu_df['datetime'])
        spread['datetime'] = spread.index
        spread.index = mu_df.index

        df = pd.DataFrame()
        df['datetime'] = main_data['datetime']
        df['signal'] = spread['spread']
        df['upbond'] = mu_df['mu'] + t * std_df['std']
        df['downbond'] = mu_df['mu'] - t * std_df['std']
        df['over'] = df[df['signal'] > df['upbond']]['signal']
        df['below'] = df[df['signal'] < df['downbond']]['signal']
        df['over'] = df['over'].apply(lambda x: True if pd.notnull(x) else x)
        df['below'] = df['below'].apply(lambda x: False if pd.notnull(x) else x)

        res_df = pd.DataFrame()
        res_df['datetime'] = main_data['datetime']
        res_df['res'] = df['over'].combine_first(df['below'])
        return res_df

    
    def get_res_df(self, util, main_data, date, delay, title):
        if title == "eff_price":
            res_df = self.eff_signal(main_data)
        elif title == "hk_signal":
            res_df = self.hk_signal(util, main_data)
        elif title == "deff_price":
            res_df = self.deriv_signal(main_data, delay)
        elif title == "dp":
            res_df = self.dp_signal(main_data)
        elif title == "div":
            res_df = self.div_deriv(main_data, delay)

        elif title == "maluyao":
            res_df = self.get_maluyao(util, main_data, date, k = 0)

        res_df['res'] = res_df['res'].shift(delay)
        return res_df
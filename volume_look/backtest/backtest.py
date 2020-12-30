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
from volume_look.structure.construct_signal import ConstructSignal
from volume_look.util.log import Logger
pd.options.mode.chained_assignment = None

class Backtest(object):
    def __init__(self, trade_at_mid, delay, title):
        self.t = 90
        self.tr_cost_buy = 0.000023
        self.tr_cost_sell = 0.000023
        # self.tr_cost_sell = 0.000345
        self.trade_at_mid = trade_at_mid
        self.delay = delay
        self.title = title

    def cal_pnl(self, util, date, ticker, morning):
        """calculate the pnl"""
        bd = ConstructSignal(morning)
        main_data = bd.build_main_data(util, date, ticker)
        res_df = bd.get_res_df(util, main_data, date, self.delay, self.title)
        main_data['index'] = main_data.index
        main_data = main_data.set_index(['datetime'], drop = False)
        main_data = main_data[~main_data.index.duplicated()]
        main_data = main_data.reindex(index = res_df['datetime'])
        main_data = main_data.set_index(['index'])

        mid_price = bd.get_mid_price(main_data)
        ask_df = bd.get_ask(main_data, mid_price, self.trade_at_mid)
        bid_df = bd.get_bid(main_data, mid_price, self.trade_at_mid)
        own = False

        n = len(res_df)
        pos, pnl = 0, 0
        tc = 0
        buy_price, sell_price = 0, 0
        strat = [0] * n
        signal_pnl = pd.DataFrame()
        signal_pnl['datetime'] = res_df['datetime']
        signal_pnl = signal_pnl.set_index(['datetime'], drop = False)
        signal_pnl['pnl'] = np.nan
        signal_pnl['signal'] = np.nan
        signal_pnl['tv'] = np.nan
        signal_pnl["returns"] = np.nan

        if n == 0:
            return strat, pnl, tc, buy_price, sell_price, signal_pnl

        ind_open = bd.get_ind_open(util, main_data, morning)
        ind_close = bd.get_ind_close(util, main_data, morning)

        for i in range(self.delay, n):
            res = res_df["res"].iloc[i]
            date_time = res_df["datetime"].iloc[i]
            if res == True and not own:
                if date_time < ind_close and date_time >= ind_open:
                    strat[i] = 1
                    own = True
                    pos = 1
                    buy_price = ask_df["ask"].iloc[i]
                    tc = buy_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "signal"] = 1
                    signal_pnl.loc[date_time, "tv"] = 1

            elif res == False and not own:
                if date_time < ind_close and date_time >= ind_open:
                    strat[i] = -1
                    own = True
                    pos = -1
                    sell_price = bid_df["bid"].iloc[i]
                    tc = sell_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "signal"] = -1
                    signal_pnl.loc[date_time, "tv"] = 1

            elif own and pos == 1 and res == False:
                strat[i] = -1
                own = False
                pos = 0
                sell_price = bid_df["bid"].iloc[i]
                tc += sell_price * self.tr_cost_sell

                diff = sell_price - buy_price - tc
                trade_pnl = diff if ((sell_price > 0) & (buy_price > 0)) else 0
                pnl += trade_pnl
                signal_pnl.loc[date_time, "pnl"] = trade_pnl
                signal_pnl.loc[date_time, "returns"] = trade_pnl / sell_price
                signal_pnl.loc[date_time, "signal"] = -1

                if date_time >= ind_open and date_time < ind_close:
                    strat[i] = -2
                    own = True
                    pos = -1
                    sell_price = bid_df["bid"].iloc[i]
                    tc = sell_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "tv"] = 1

            elif own and pos == -1 and res == True:
                strat[i] = 1
                own = False
                pos = 0
                buy_price = ask_df["ask"].iloc[i]
                tc += buy_price * self.tr_cost_sell
                diff = sell_price - buy_price - tc
                trade_pnl = diff if ((sell_price > 0) & (buy_price > 0)) else 0
                pnl += trade_pnl
                signal_pnl.loc[date_time, "pnl"] = trade_pnl
                signal_pnl.loc[date_time, "returns"] = trade_pnl / buy_price
                signal_pnl.loc[date_time, "signal"] = 1

                if date_time >= ind_open and date_time < ind_close:
                    strat[i] = 2
                    own = True
                    pos = 1
                    buy_price = ask_df["ask"].iloc[i]
                    tc = buy_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "tv"] = 1
        return (
            strat,
            pnl,
            tc,
            buy_price,
            sell_price,
            signal_pnl,
        )    

    def wind_up(self, util, date, ticker, morning, strat, tc, buy_price, sell_price, pnl, signal_pnl):
        """wind trade up"""
        signal_delay = util.get_signal_delay()
        n = len(strat) - 1
        bd = ConstructSignal(morning)
        main_data = bd.build_main_data(util, date, ticker)
        mid_price = bd.get_mid_price(main_data)
        ask = bd.get_ask(main_data, mid_price, self.trade_at_mid)
        bid = bd.get_bid(main_data, mid_price, self.trade_at_mid)

        if sum(strat) == 1:
            if strat[n] == 1:
                strat[n] = 0
                signal_pnl["tv"].iloc[n] = 0
            else:
                strat[n] = -1
                sell_price = bid["bid"].iloc[n]
                tc += sell_price * self.tr_cost_sell
                diff = sell_price - buy_price - tc
                trade_pnl = diff if ((sell_price > 0) & (buy_price > 0)) else 0
                pnl += trade_pnl
                signal_pnl["pnl"].iloc[n] = trade_pnl
                signal_pnl["returns"].iloc[n] = trade_pnl / sell_price
                signal_pnl["signal"].iloc[n] = -1

        elif sum(strat) == -1:
            if strat[n] == -1:
                strat[n] = 0
                signal_pnl["tv"].iloc[n] = 0
            else:
                strat[n] = 1
                buy_price = ask["ask"].iloc[n]
                tc += buy_price * self.tr_cost_sell
                diff = sell_price - buy_price - tc
                trade_pnl = diff if ((sell_price > 0) & (buy_price > 0)) else 0
                pnl += trade_pnl
                signal_pnl["pnl"].iloc[n] = trade_pnl
                signal_pnl["returns"].iloc[n] = trade_pnl / buy_price
                signal_pnl["signal"].iloc[n] = 1
        return pnl, signal_pnl
    
    def operate_strategy(self, util, date):
        """operate the strategy"""
        print(date)
        cs = ConstructSignal(True)
        ticker = cs.get_main_ticker(util, date)
        if ticker is None:
            return 0, pd.DataFrame()
        (
            strat,
            pnl,
            tc,
            buy_price,
            sell_price,
            signal_pnl,
        ) = self.cal_pnl(util, date, ticker, True)
        pnl_m, signal_pnl_m = self.wind_up(
            util,
            date,
            ticker,
            True,
            strat,
            tc,
            buy_price,
            sell_price,
            pnl,
            signal_pnl
        )
        (
            strat,
            pnl,
            tc,
            buy_price,
            sell_price,
            signal_pnl
        ) = self.cal_pnl(util, date, ticker, False)
        pnl_e, signal_pnl_e = self.wind_up(
            util,
            date,
            ticker,
            False,
            strat, 
            tc,
            buy_price,
            sell_price,
            pnl,
            signal_pnl
        )

        pnl = pnl_m + pnl_e
        signal_pnl = signal_pnl_m.append(signal_pnl_e)
        return pnl, signal_pnl

    

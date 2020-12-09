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
from volume_look.util.save_util import SaveUtil
from volume_look.analysis.build_data import BuildData

class BackTest(object):
    def __init__(self, trade_at_mid):
        self.t = 90
        self.tr_cost_buy = 0.000023
        self.tr_cost_sell = 0.000023
        # self.tr_cost_sell = 0.000345
        self.trade_at_mid = trade_at_mid

    def cal_pnl(self, util, date, ticker, morning, res_df):
        bd = BuildData()
        main_data = bd.build_main_data(util, date, ticker)
        mid_price = bd.get_mid_price(main_data)
        ask_df = bd.get_ask(main_data, mid_price, self.trade_at_mid)
        bid_df = bd.get_bid(main_data, mid_price, self.trade_at_mid)
        own = False

        n = len(res_df)
        pos = 0
        strat = [0] * n
        signal_pnl = pd.DataFrame()
        signal_pnl['datetime'] = res_df['datetime']
        signal_pnl = signal_pnl.set_index(['datetime'], drop = False)
        signal_pnl['pnl'] = np.nan
        signal_pnl['signal'] = np.nan
        signal_pnl['tv'] = np.nan
        signal_pnl["returns"] = np.nan

        ind_open = bd.get_ind_open(util, main_data, morning)
        ind_close = bd.get_ind_close(util, main_data, morning)

        for i in range(0, n):
            res = efpc_vec_df["res"].iloc[i]
            date_time = efpc_vec_df["datetime"].iloc[i]
            if res and not own:
                if date_time < ind_close and date_time >= ind_open:
                    strat[i] = 1
                    own = True
                    pos = 1
                    buy_price = ask_df["ask"].iloc[i]
                    tc = buy_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "signal"] = 1
                    signal_pnl.loc[date_time, "tv"] = 1

            elif not res and not own:
                if date_time < ind_close and date_time >= ind_open:
                    strat[i] = -1
                    own = True
                    pos = -1
                    sell_price = bid_df["bid"].iloc[i]
                    tc = sell_price * self.tr_cost_buy
                    signal_pnl.loc[date_time, "signal"] = -1
                    signal_pnl.loc[date_time, "tv"] = 1

            elif own and pos == 1 and not res:
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

            elif own and pos == -1 and res:
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

    def wind_up(self, util, date, ticker, morning, strat, buy_price, sell_price, pnl, signal_pnl):
        signal_delay = util.get_signal_delay()
        n = len(strat) - 1
        bd = BuildData()
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
        res = 
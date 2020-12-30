# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import traceback

import statsmodels.api as sm
from volume_look.util.utility import Utility
from volume_look.util.save_util import SaveUtil
from volume_look.util.log import Logger
from volume_look.structure.construct_signal import ConstructSignal
from volume_look.backtest.backtest import Backtest


class SignalRun(object):
    def __init__(
        self, start_date, end_date, config, trade_at_mid, title
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.trade_at_mid = trade_at_mid
        self.title = title

    def start(self, ss):
        """sweet strategy start"""
        logger = Logger(
            os.path.join(self.config["prefix"], self.config["files_path"]["log"])
        )

        util = Utility(self.config)
        su = SaveUtil(self.config)
        # util.create_dirs()
        logger.warn("{self.title}".format_map(vars()))
        date_lst = util.generate_date_lst(self.start_date, self.end_date)
        logger.warn(date_lst[0])
        logger.warn(date_lst[-2])
        pnl_lst = []
        signal_df = pd.DataFrame()
        lags = util.get_time_lag()
        delay = util.get_delay()

        for i in range(0, len(date_lst) - 1):
            date = date_lst[i]
            log_context = str(date) + " " + str(self.title)
            logger.info(log_context)
            pnl, signal = ss.operate_strategy(util, date)
            pnl_lst.append(pnl)
            signal_df = signal_df.append(signal)

        su.save_signal_signal(signal_df, date_lst, self.title, self.trade_at_mid, lags, delay)
        su.save_signal_pnl(pnl_lst, date_lst, self.title, self.trade_at_mid, lags, delay)
        logger.warn('complete')
        return pnl_lst

    def trigger(self):

        util = Utility(self.config)
        # util.create_dirs()
        delay = util.get_delay()
        ss = Backtest(self.trade_at_mid, delay, self.title)
        try:
            self.start(ss)
        except Exception as e:
            logger = Logger(
            os.path.join(self.config["prefix"], self.config["files_path"]["log"])
            )
            logger.error(str(traceback.format_exc()))


def main():
    config_path = "/home/lky/volume_speculate/volume_look/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    start_date = "20200101"
    end_date = "20200301"
    trade_at_mid = False

    # title = "eff_price"
    # title = "deff_price"
    # title = "hk_signal"
    # title = "dp"
    title = "maluyao"
    # print(trade_at_mid)
    tr = SignalRun(
        start_date, end_date, config, trade_at_mid, title
    )
    tr.trigger()



if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, distinct, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from cachetools import cached, TTLCache
from dzapi import __version__
from dzapi.model import *
from volume_look.util.log import Logger
import pandas as pd
import numpy as np
import os
import datetime
import collections
import calendar

class SaveUtil(object):
    def __init__(self, config):
        self.config = config
    
    def save_resid(self, df, level, date):
        path = os.path.join(self.config['prefix'], self.config['dir_path']['pair'], 
                        self.config['dir_path']['resid'], str(level), str(date) + '.csv')
        df.to_csv(path, header = True)
    
    def save_coef(self, lst, level, date_lst, remain_far, remain_close):
        df = pd.DataFrame()
        df['coef'] = lst
        df['date'] = date_lst
        df['remain_close'] = remain_close
        df['remain_far'] = remain_far

        start_date, end_date = date_lst[0], date_lst[-1]
        path = os.path.join(self.config['prefix'], self.config['dir_path']['pair'], 
                        self.config['dir_path']['coef'], str(start_date) + '_' + str(end_date) + '_' + str(level) + '.csv')
        df.to_csv(path, header = True)
    
    def save_ratio(self, lst, level, date_lst, remain_far, remain_close, ticker_lst):
        df = pd.DataFrame()
        df['ratio'] = lst
        df['date'] = date_lst
        df['remain_far'] = remain_far
        df['remain_close'] = remain_close
        df['ticker'] = ticker_lst

        start_date, end_date = date_lst[0], date_lst[-1]
        path = os.path.join(self.config['prefix'], self.config['dir_path']['pair'], 
                        self.config['dir_path']['ratio'], str(start_date) + '_' + str(end_date) + '_' + str(level) + '.csv')
        df.to_csv(path, header = True)
    
    def save_spot_futures(self, df, level, date, remain, ticker):
        df['remain'] = remain
        df['ticker'] = ticker

        path = os.path.join(self.config['prefix'], self.config['dir_path']['pair'], 
                        self.config['dir_path']['spot_futures'], str(level), str(date) + '.csv')
        df.to_csv(path, header = True)

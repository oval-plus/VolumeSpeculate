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

class Utility(object):
    def __init__(self, config):
        self.config = config

    def get_wind_engine(self):
        """get the wind engine"""
        wind_engine = create_engine(self.config['wind_engine'], poolclass = NullPool)
        return wind_engine
    
    def get_trade_days(self, start_date, end_date):
        """get the trade days between the start_date and end_date"""
        wind_engine = self.get_wind_engine()
        Session = sessionmaker(bind=wind_engine)
        session = Session()
        
        query_date = session.query(ASHARECALENDAR.TRADE_DAYS) \
            .filter(ASHARECALENDAR.TRADE_DAYS <= end_date) \
            .filter(ASHARECALENDAR.TRADE_DAYS >= start_date) \
            .filter(ASHARECALENDAR.S_INFO_EXCHMARKET == 'SSE').statement
        session.close()
        return query_date
    
    def get_sql_calendar(self, end_date):
        """get the trade days before the end_date"""
        wind_engine = self.get_wind_engine()
        Session = sessionmaker(bind=wind_engine)
        session = Session()
        
        query_date = session.query(ASHARECALENDAR.TRADE_DAYS) \
            .filter(ASHARECALENDAR.TRADE_DAYS <= end_date) \
            .filter(ASHARECALENDAR.S_INFO_EXCHMARKET == 'SSE').statement
        session.close()
        return query_date
    
    def get_trade_calendar(self, end_date):
        """get the dataframe of the trade days before the end_date"""
        result = self.get_sql_calendar(end_date)
        wind_engine = self.get_wind_engine()
        df = pd.read_sql(result, con = wind_engine)
        new = df['TRADE_DAYS'].sort_values(ascending = True)
        new = new.reset_index(drop = True)
        return new
    
    def get_back_date(self, date):
        """get the day before the specify date"""
        df = self.get_trade_calendar(date)
        k = -2
        old_date = df.iloc[k]
        data = self.open_data(old_date)
        while data.empty:
            k -= 1
            old_date = df.iloc[k]
            data = self.open_data(old_date)
        
        return old_date
    
    def get_week_date(self, date, k):
        """get a certain k date before the specify date"""
        df = self.get_trade_calendar(date)
        k = -k
        old_date = df.iloc[k]
        data = self.open_data(old_date)
        while data.empty:
            k -= 1
            old_date = df.iloc[k]
            data = self.open_data(old_date)
        return old_date

    def get_price(self, df, time):
        """get the next price"""
        if df['sod'].iloc[-1] >= time:
            time = df[df.sod >= time].iloc[0]
        else:
            time = df.iloc[-5]
        return time

    def generate_date_lst(self, start_date, end_date):
        trade_days_df = self.get_tradedays(start_date, end_date)
        date_lst = trade_days_df.tolist()
        return date_lst

    def get_ticker_maturity(self, ticker):
        year, month = ticker[2: 4], ticker[4: 6]
        prefix = '20' + year + month
        monthrange = calendar.monthrange(int(year), int(month))[1]
        start_date, end_date = prefix + '01', prefix + str(monthrange)
        maturity_df = self.get_expiration_date(start_date, end_date)
        maturity_date = maturity_df[maturity_df['res'] == True].index[0]
        return maturity_date

    def generate_expiration_lst(self, date_lst):
        """generate expiration date list"""
        start_date, end_date = date_lst[0], date_lst[-1]
        expiration_df = self.get_expiration_interval(start_date, end_date)
        expiration_old, expiration_new = [], []
        for i in range(0, len(date_lst) - 1):
            date = date_lst[i]
            new_date = date_lst[i + 1]
            expiration_old.append(expiration_df['res'].loc[date])
            expiration_new.append(expiration_df['res'].loc[new_date])
        return expiration_old, expiration_new

    def get_tradedays(self, start, end):
        """get the trade days between start and end within the dataframe form"""
        result = self.get_trade_days(start, end)
        wind_engine = self.get_wind_engine()
        df = pd.read_sql(result, con = wind_engine)
        new = df['TRADE_DAYS'].sort_values(ascending = True)
        new = new.reset_index(drop = True)
        return new
    
    def get_expiration_date(self, start_date, end_date):
        """determine whether it is the expiration date"""
        trade_day_df = self.get_tradedays(start_date, end_date)
        month_dict = collections.defaultdict(int)
        mask_df = pd.DataFrame(columns = ['date', 'res'])
        mask_df['date'] = trade_day_df

        for i in range(0, len(mask_df)):
            month = trade_day_df.iloc[i][4: 6]
            date = datetime.datetime.strptime(trade_day_df.iloc[i][:10], '%Y%m%d')
            mask_df['res'].iloc[i] = False
            if date.weekday() == 4:
                if 15 <= date.day <= 21:
                    mask_df['res'].iloc[i] = True
                    month_dict[month] = True
                elif 22 <= date.day <= 28:
                    if not month_dict.get(month, False):
                        month_dict[month] = True
                        mask_df['res'].iloc[i] = True
        mask_df = mask_df.set_index(['date'])
        return mask_df

    def get_expiration_interval(self, start_date, end_date):
        """get the expiration interval
        
        Args:
            end_date: the end date of the trade day
        """
        mask_df = self.get_expiration_date(start_date, end_date)
        month_dict = collections.defaultdict(int)
        mask_df['second_friday'] = None
        for i in range(0, len(mask_df)):
            month = mask_df.index[i][4: 6]
            if mask_df['res'].iloc[i]:
                if i >= 5:
                    mask_df['second_friday'].iloc[i - 5] = True
        mask_df = mask_df.fillna(False)

        month_dict = dict()
        idx = mask_df[mask_df.second_friday | mask_df.res].index
        df = pd.DataFrame()
        df['date'] = mask_df.index
        df = df.set_index(['date'])
        df['res'] = None
        df = df.reindex(idx).fillna(True)
        df = df.reindex(mask_df.index).fillna(False)
        
        for i in range(0, len(mask_df)):
            month = mask_df.index[i][4: 6]
            if df['res'].iloc[i] and not month_dict.get(month, False):
                month_dict[month] = True
            elif df['res'].iloc[i] and month_dict.get(month, False):
                month_dict[month] = False
            elif month_dict.get(month, False) and not df['res'].iloc[i]:
                df['res'].iloc[i] = True
        # df = df.reindex(_idx)
        return df

    def hdf_csv(self, path, date):
        """hdf to csv"""
        with pd.HDFStore(path) as hdf:
            lst = hdf.keys()
        whole_df = pd.DataFrame()
        for contract in lst:
            table_ori = pd.read_hdf(path, key = contract)
            table_ori = table_ori.reset_index(drop = False)
            table_ori.columns = ['datetime', 'ticker', 'cp', 's1', 'b1', 'sv1', 'bv1', 'hp', 'lp', 'ts', 'tt', 'oi']
            table_ori['updatetime'] = table_ori['datetime'].apply(lambda x: str(x)[11:])
            table_ori['updatetime'] = pd.to_datetime(table_ori['updatetime'], format = '%H:%M:%S.%f')
            table_ori['sod'] = table_ori['updatetime'].apply(
                lambda x: datetime.timedelta(hours = x.hour, minutes=x.minute, seconds=x.second).total_seconds())
            
            whole_df = whole_df.append(table_ori, ignore_index = True)

        new_path = os.path.join(self.config['read_prefix'], self.config['dir_path']['quote'],
                str(date) + '.csv')
        whole_df.to_csv(new_path)
        return whole_df

    def open_data(self, date):
        """open the data"""
        path = os.path.join(self.config['read_prefix'], self.config['dir_path']['quote'])
        file_path = os.path.join(path, str(date) + '.csv')
        if not os.path.exists(file_path):
            h5_path = os.path.join(self.config['read_prefix'], self.config['dir_path']['h5'], str(date) + '.h5')
            if not os.path.exists(h5_path):
                raise Exception(str(date) + '.h5' + ' does not exist, please check the dir path or the file path.')
            data = self.hdf_csv(h5_path, date)
        else:
            data = pd.read_csv(file_path, index_col = 0)
        if data.empty:
            return data
        else:
            contract_type = self.config['contract_type']
            data = data[data['ticker'].str.contains(contract_type)]
        return data
    
    def get_open_time(self, date, morning)  :
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            if morning:
                open_time = self.config['shen_time']['new_AM']['open']
            else:
                open_time = self.config['shen_time']['new_PM']['open']
        else:
            if morning:
                open_time = self.config['shen_time']['AM']['open']
            else:
                open_time = self.config['shen_time']['PM']['open']
        return open_time

    def get_close_time(self, date, morning):
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            if morning:
                open_time = self.config['shen_time']['new_AM']['close']
            else:
                open_time = self.config['shen_time']['new_PM']['close']
        else:
            if morning:
                open_time = self.config['shen_time']['AM']['close']
            else:
                open_time = self.config['shen_time']['PM']['close']
        return open_time        

    def get_signal(self, weekday, week_signal):
        """get the week effect signal"""
        if ((week_signal == -1 and (weekday == 0 or weekday == 3)) or 
            (week_signal == 1 and (weekday == 1 or weekday == 2))):
            signal = -1
        elif ((week_signal == 1 and (weekday == 0 or weekday == 4)) or 
            (week_signal == -1 and (weekday == 1 or weekday == 2))):
            signal = 1
        else:
            signal = 0
        return signal
    
    def get_start_time(self, date):
        """get the start time
            开市时间
        Args:
            date: the date
        """
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            start_time = self.config['new_rule']['start']
        else:
            start_time = self.config['rule']['start']
        return start_time
    
    def get_end_time(self, date):
        """get the end time"""
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            end_time = self.config['new_rule']['end']
        else:
            end_time = self.config['rule']['end']
        return end_time
    
    def _get_tail_time(self, date):
        """尾盘涨幅/买卖单不平衡度，收市前半小时"""
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            end_time = self.config['new_rule']['tail_timing']
        else:
            end_time = self.config['rule']['tail_timing']
        return end_time
    
    def buy_time_last(self):
        tail_time = self.config['tail'] * 60 + 2
        return tail_time
    
    def get_sell_time(self, date):
        """翌日上午十点卖出"""
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            end_time = self.config['new_rule']['sell_timing']
        else:
            end_time = self.config['rule']['sell_timing']
        return end_time
    
    def get_buy_datetime(self, date):
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            end_time = self.config['new_rule']['buy_datetime']
        else:
            end_time = self.config['rule']['buy_datetime']
        return end_time
    
    def get_sell_datetime(self, date):
        specify_date = self.config['rule_date']
        if datetime.datetime.strptime(date[:10], '%Y%m%d') >= datetime.datetime.strptime(specify_date, '%Y%m%d'):
            end_time = self.config['new_rule']['sell_datetime']
        else:
            end_time = self.config['rule']['sell_datetime']
        return end_time

    def get_threshold(self):
        threshold = self.config['parameters']['threshold']
        return threshold
    
    def deter_choice(self, choice):
        if choice == 'two':
            return 'intersection'
        elif choice == 'interaction':
            return 'intersection'
        elif choice == 'all':
            return 'three'
        elif choice == 'price':
            return 'price_gap'
        
        choice_lst = ['intersection', 'three', 'price_gap', 'union', 
                'imbalance', 'basis', 'tail']
        if choice not in choice_lst:
            raise Exception('choice is not legal, please check your strategy choice.')
        else:
            return choice

    def get_stock_price(self, date):
        """现货的价格，以分钟为单位"""
        path = os.path.join(self.config['read_prefix'], self.config['dir_path']['stock_price'])
        path = os.path.join(path, self.config['files_path']['IF'])
        df = pd.read_csv(path, index_col = 0)
        df.index = df.index.astype(str)
        
        date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        df = df[df.index.str.contains(date)]
        return df
    
    def get_close_price(self, date, ticker):
        """get the close price of the index futures in the float type"""
        path = os.path.join(self.config['read_prefix'], self.config['dir_path']['settle_close'])
        path = os.path.join(path, str(ticker) + '.csv')
        df = pd.read_csv(path, index_col = 0)
        df.index = df.index.astype(str)
        return df['CLOSE'].loc[date]

    def get_settle_price(self, date, ticker):
        path = os.path.join(self.config['read_prefix'], self.config['dir_path']['settle_close'])
        path = os.path.join(path, str(ticker) + '.csv')
        df = pd.read_csv(path, index_col = 0)
        df.index = df.index.astype(str)
        return df['SETTLE'].loc[date]
    
    def get_settle_close(self, date, ticker):
        path = os.path.join(self.config['read_prefix'], self.config['dir_path']['settle_close'])
        path = os.path.join(path, str(ticker) + '.csv')
        df = pd.read_csv(path, index_col = 0)
        df.index = df.index.astype(str)
        return df
    
    def get_signal_delay(self):
        delay = 0
        return delay
    
    def last_trade_time(self):
        last_time = self.config['last_trade_time']
        return last_time

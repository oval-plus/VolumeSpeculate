# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from scipy import optimize
from functools import partial
from volume_look.util.utility import Utility

config_path = r"..."
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)
start_date = '20191001'
end_date = '20200201'
date_lst = util.generate_date_lst(start_date, end_date)

df = pd.read_csv(r'df_20190101_20200401.csv', index_col = 0)
k = 2
u_lst = []
omega_lst = []

time_0 = "00:00:00.000"
time_1 = "09:40:00.000"
time_2 = "10:00:00.000"
time_3 = "10:10:00.000"

def cons_theta_s(x, num):
    return x[0] - num

def cons_theta_l(x, num):
    return -(x[0] - num)

def cons_omega_s(x, num):
    return x[1] - num

def cons_omega_l(x, num):
    return -(x[1] - num)

def con_func(args):
    con1 = {'type': 'ineq', 'fun': partial(cons_theta_s, num = args[0])}
    con2 = {"type": "ineq", "fun": partial(cons_theta_l, num = args[0])}
    con3 = {"type": "ineq", "fun": partial(cons_omega_s, num = args[1])}
    con4 = {"type": "ineq", "fun": partial(cons_omega_l, num = args[1])}
    return [con1, con2, con3, con4]


def datefmt_convert(date):
    return datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')

for i in range(0, len(date_lst)):
    date = date_lst[i]
    check_time_0 = datefmt_convert(date) + ' ' + time_0
    check_time_1 = datefmt_convert(date) + ' ' + time_1
    check_time_2 = datefmt_convert(date) + ' ' + time_2
    check_time_3 = datefmt_convert(date) + ' ' + time_3

    df['date'] = df['date'].apply(str)
    df_0_1 = df[(df['datetime'] >= check_time_0) & (df['datetime'] <= check_time_1)]
    df_1_2 = df[(df['datetime'] >= check_time_1) & (df['datetime'] <= check_time_3)]
    df_0_2 = df[(df['datetime'] >= check_time_0) & (df['datetime'] <= check_time_3)]

    mu = np.mean(df_0_1['spread'])
    sigma = np.std(df_0_1['spread'])
    X = df_1_2[np.abs(df_1_2['spread'] - mu) < k*sigma]
    X_mean = np.mean(X[(X['datetime']>=check_time_2)&(X['datetime']<=check_time_3)]['spread'])
    X['spread'] = X['spread'] - X_mean
    X['last_spread'] = X['spread'].shift(1)
    #MLE:bound constrained optimization
    args1 = (0.01,0.9, 0.01,0.9)
    cons = (con_func(args1))
    X2 = (X['spread'].astype(float))
    X1 = (X['last_spread'].astype(float))
    def log_likelihood(params):
        theta,omega = params
        delta = 1 / 250 /241
        omega2 = omega*np.sqrt((1 - np.exp(-2*theta*delta))/(2*theta))
        N = 241 * 31-1
        L = -1 * (-N / 2 * np.log(2 * np.pi) - N * np.log(omega2) - np.sum(1 / (2 *
        omega2 ** 2)*(X2 - X1 * np.exp(-theta * delta)) ** 2))
        return L
    params = [0.00001, 0.5]
    # theta_omega=optimize.minimize(log_likelihood,x0=params,
    # method='SLSQP',constraints=cons)
    theta_omega = optimize.minimize(log_likelihood, x0 = params)
    theta = theta_omega.x[0] 
    omega = theta_omega.x[1]
    gt = np.mean(X.iloc[:-1].spread)
    gt_last = np.mean(X.iloc[1:].spread)
    u = 1 / theta * (gt - gt_last) + gt
    u_lst.append([date, u])
    omega_lst.append([date, np.std(X.spread)])

u_df = pd.DataFrame(u_lst)
u_df.columns = ['date','u']
omega_df = pd.DataFrame(omega_lst)
omega_df.columns = ['date', 'omega']
df1 = pd.merge(df, u_df)
df1 = pd.merge(df1, omega_df)
df1['up'] = df1['u'] + k * df1['omega']
df1['down'] = df1['u'] - k * df1['omega']

df1['holding_close'] = 0
df1['holding_far'] = 0

up_flag = 0
down_flag = 0
n = len(df1)
for i in range(1, n):
    if df1['updatetime'].iloc[i][11:] < time_3:
        continue
    if df1['spread'].iloc[i] > df1['up'].iloc[i]:
        if not up_flag:
            df1['holding_close'].iloc[i] = 1
            df1['holding_far'].iloc[i] = -1
            up_flag = 1
            down_flag = 0
        continue
    if df1['spread'].iloc[i] < df1['down'].iloc[i]:
        if not down_flag:
            df1['holding_close'].iloc[i] = -1
            df1['holding_far'].iloc[i] = 1
            down_flag = 1
            up_flag = 0

df1['mid_price'] = (df1['s1'] + df1['b1']) / 2
df1['portfolio_ret'] = df1['holding_close'] * df1['mid_price'] + \
    df1['holding_far'] * df1['mid_price_2']

trading_part = df1[df1['holding_close'] != 0]
trading_part['commision'] = trading_part['mid_price'] * 0.000023
trading_part['commision_2'] = trading_part['mid_price_2'] * 0.000023
trading_part['mul'] = 0.000023

n = len(trading_part)
flag_date = ''
for i in range(0, n):
    
    if trading_part['date'].iloc[i] == flag_date:
        k += 1
        if k % 2:
            trading_part['commision'].iloc[i] = trading_part['mid_price'].iloc[i] * 0.000345
            trading_part['commision_2'].iloc[i] = trading_part['mid_price_2'].iloc[i] * 0.000345
            trading_part['mul'].iloc[i] = 0.000345
    else:
        k = 0
    flag_date = trading_part['date'].iloc[i]

real_pnl = pd.DataFrame()
real_pnl['pnl'] = trading_part['portfolio_ret'] - trading_part['commision'] - trading_part['commision_2']
real_pnl['date'] = trading_part['date']
real_pnl['pnl_cumsum'] = real_pnl['pnl'].cumsum()

print(real_pnl['pnl'].sum())

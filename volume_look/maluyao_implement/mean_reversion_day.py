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
from volume_look.util.utility import Utility

config_path = r"..."
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)
start_date = '20191001'
end_date = '20200201'
date_lst = util.generate_date_lst(start_date, end_date)

df = pd.read_csv(r'df_beside.csv', index_col = 0)
k = 2
u_lst = []
omega_lst = []

time_0 = "00:00:00.000"
time_1 = "09:10:00.000"
time_2 = "09:35:00.000"
time_3 = "09:40:00.000"

for i in range(0, len(date_lst)):
    date = date_lst[i]
    check_time_0 = date + ' ' + time_0
    check_time_1 = date + ' ' + time_1
    check_time_2 = date + ' ' + time_2
    check_time_3 = date + ' ' + time_3

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
    # cons = con(args1)
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

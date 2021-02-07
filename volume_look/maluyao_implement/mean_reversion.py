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
end_date = '20200401'
datelist = util.generate_date_lst(start_date, end_date)


def gather_data(util, date_lst):
    whole_df = pd.DataFrame()
    for i in range(0, len(date_lst)):
        date = date_lst[i]
        _df = util.open_data(date)
        _df['datetime'] = _df['datetime'].apply(
                lambda x: x[:-3] + '000' if x[-3] < "5" else x[:-3] + '500')
        if _df.empty:
            continue
        ticker_lst = _df[~_df['ticker'].duplicated()]['ticker'].tolist()
        df = _df[_df['ticker'] == ticker_lst[0]]
        df['is_overnight'] = np.nan
        df['is_overnight'].iloc[0] = 1
        df['is_overnight'].iloc[1] = 1
        
        df['date'] = date
        df = df[~df['datetime'].duplicated()]

        # future_2 = _df[_df['ticker'] == ticker_lst[2]]
        future_2 = _df[_df['ticker'] == ticker_lst[1]]
        df['spread'] = cal_extra_spread(df, future_2, date)['spread']
        new_cp = _df[_df['ticker'] == ticker_lst[1]]
        new_cp = new_cp[~new_cp['datetime'].duplicated()]
        new_cp = new_cp.set_index(['datetime'])
        new_cp = new_cp.reindex(index = df['datetime'])
        new_cp.index = df.index
        # df['cp_2'] = new_cp['cp']
        df['mid_price_2'] = (new_cp['s1'] + new_cp['b1']) / 2
        whole_df = whole_df.append(df)
        print(date)
    return whole_df

def cal_extra_spread(futures_1, futures_2, date):
    trade_1 = futures_1['cp'].iloc[0]
    trade_2 = futures_2['cp'].iloc[0]

    # ln_1 = futures_1['cp'].apply(lambda x: np.log(x / trade_1))
    # ln_2 = futures_2['cp'].apply(lambda x: np.log(x / trade_2))
    ln_1 = futures_1['cp'].apply(lambda x: np.log(x))
    ln_2 = futures_2['cp'].apply(lambda x: np.log(x))
    
    ln_1.index = futures_1['datetime'].apply(lambda x: str(x)[11:])
    ln_2.index = futures_2['datetime'].apply(lambda x: str(x)[11:])

    spread = ln_1 - ln_2
    
    date = futures_1['date'].iloc[0]
    check_date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    spread_df = pd.DataFrame()
    spread_df['spread'] = spread
    spread_df['datetime'] = spread.index.map(lambda x: check_date + " " + x)

    
    # spread_df['datetime'] = futures_1['datetime']
    spread_df = spread_df.set_index(['datetime'])
    spread_df = spread_df[~spread_df.index.duplicated()]
    spread_df = spread_df.reindex(index = futures_1['datetime'])
    spread_df.index = futures_1.index
    # spread_df.index = futures_1.index
    return spread_df

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
# df = gather_data(util, datelist)
# df.to_csv('df_beside.csv')



df = pd.read_csv(r'...', index_col = 0)
k = 2
ulist = []
omegalist = []
theta_lst = []
for i in range(10, len(datelist)):
    i0 = i - 10
    i1 = i0 + 5
    i2 = i - 3
    df['date'] = df['date'].apply(str)
    df_0_1 = df[(df['date']>=datelist[i0]) & (df['date']<=datelist[i1])]
    df_1_2 = df[(df.date>=datelist[i1])&(df.date<=datelist[i])]
    df_0_2 = df[(df.date>=datelist[i0])&(df.date<=datelist[i])]
    overnight_df_0_1 = df_0_1[df_0_1.is_overnight.astype(float)==1]
    mu = np.mean(overnight_df_0_1['spread'])
    sigma = np.std(overnight_df_0_1['spread'])
    X = df_1_2[np.abs(df_1_2['spread'] - mu) < k*sigma]
    X_mean = np.mean(X[(X['date']>=datelist[i2])&(X['date']<=datelist[i])]['spread'])
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
    theta_omega=optimize.minimize(log_likelihood,x0=params,
        method='SLSQP',constraints=cons)
    # theta_omega = optimize.minimize(log_likelihood, x0 = params)
    theta = theta_omega.x[0]
    omega = theta_omega.x[1]
    theta_lst.append([theta, omega])
    gt = np.mean(X.iloc[:-1].spread)
    gt_last = np.mean(X.iloc[1:].spread)
    u = 1 / theta * (gt - gt_last) + gt
    ulist.append([datelist[i],u])
    omegalist.append([datelist[i],np.std(X.spread)])



u_df = pd.DataFrame(ulist)
u_df.columns = ['date','u']
omega_df = pd.DataFrame(omegalist)
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
# df1['commision'] = df1['mid_price'] * 0.000345
# df1['commision_2'] = df1['mid_price_2'] * 0.000345

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

plot_realpnl = real_pnl[~real_pnl['date'].duplicated()]
plot_realpnl.index = pd.to_datetime(plot_realpnl['date'], format = '%Y%m%d')


def datefmt_convert(date):
    return datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
pnl_df = pd.read_csv(r'...', index_col = 0)
start_date_plot = datelist[10]
pnl_df = pnl_df[(pnl_df['date'] >= datefmt_convert(start_date_plot)) & (pnl_df['date'] <= datefmt_convert(end_date))]
pnl_df['pnl'] = pnl_df['diff'].cumsum()
pnl_nodu = pnl_df[~pnl_df['date'].duplicated()]
# pnl_nodu.index = pnl_nodu['date']
pnl_nodu.index = pd.to_datetime(pnl_nodu['date'])
pnl_nodu['real_pnl'] = plot_realpnl['pnl_cumsum']

fig, ax = plt.subplots()
ax.plot(pnl_nodu['pnl'])
ax.plot(pnl_nodu['real_pnl'])
fig.autofmt_xdate()


mask_df = pd.DataFrame()
mask_df['pnl'] = df1['portfolio_ret'] - df1['commision'] - df1['commision_2']
mask_df['date'] = df1['date']
last = df1[~df1['ticker'].duplicated(keep='last')].index
far_1 = df1['holding_far'].iloc[0: last[0]].sum()
far_2 = df1['holding_far'].iloc[last[0]: last[1] + 1].sum()
far_3 = df1['holding_far'].iloc[last[1]:].sum()

close_1 = df1['holding_close'].iloc[0: last[0]].sum()
close_2 = df1['holding_close'].iloc[last[0]: last[1] + 1].sum()
close_3 = df1['holding_close'].iloc[last[1]:].sum()

mask_df['pnl'].sum() - abs(close_1) * df1['cp'].iloc[last[0]] * (1 + 0.000345) - abs(
    close_2 + far_1) * df1['cp'].iloc[last[1]] * (1 + 0.000345) - abs(
        close_3 + far_2) * df1['cp'].iloc[last[2]] * (1 + 0.000345)
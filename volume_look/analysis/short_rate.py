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

a = 4042.5

b = 9.944893636

r = 100.5465287

def process_data(long_mean, mean_rvs, vol):
    data = pd.read_csv(r'E:\Document\high frequency\settle_close\IF2003.csv', index_col = 0)
    data = data.dropna(axis = 0, how = 'any')
    pct_chg = data.pct_change()
    
    df = pd.DataFrame()
    df['interest_rate'] = pct_chg['SETTLE']
    df['intst_rate_drift'] = df['interest_rate'].pct_change()
    df['gap'] = long_mean * (mean_rvs - df['interest_rate'])
    df['error'] = df['intst_rate_drift'] - df['gap']
    df['pdf'] = df['error'].apply(lambda x: normpdf(x, 0, vol))
    df['ln_pdf'] = df['pdf'].apply(np.log)
    return df

def normpdf(x, mu, sigma):
    u = (x-mu)/sigma
    y = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-u*u/2)
    return y


long_mean = -0.0001
mean_reversion_spd = 1
volatility = 10
df = process_data(long_mean, mean_reversion_spd, volatility)

def get_data():
    data = pd.read_csv(r'E:\Document\high frequency\settle_close\IF2003.csv', index_col = 0)
    data = data.dropna(axis = 0, how = 'any')
    pct_chg = data.pct_change()
    
    df = pd.DataFrame()
    df['interest_rate'] = pct_chg['SETTLE']
    df['intst_rate_drift'] = df['interest_rate'].pct_change()
    return df

def objective(x):
    df = get_data()
    gap = x[0] * (x[1] - df['interest_rate'])
    error = df['intst_rate_drift'] - gap
    vol = x[2]
    pdf = error.apply(lambda x: normpdf(x, 0, vol))
    ln_pdf = pdf.apply(np.log)
    return ln_pdf.sum()

def constraints(x):
    return x[2] - 0.0001

con = {'type': 'ineq', 'fun': constraints}
cons = ([con])
x0 = np.array([-0.0001, 1, 10])
solution = optimize.minimize(objective, x0, method = 'SLSQP',
            constraints = cons)


def vasicek(r0, a, b, sigma, N=10000, seed=777):    
    np.random.seed(seed)
    dt = 1/float(N)    
    rates = [r0]
    for i in range(N):
        # dr = a*dt*(b-rates[-1]) + sigma*np.random.normal()
        dr = a * (b - rates[-1]) * dt + sigma * np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

df = pd.read_csv(r'E:\Document\high frequency\settle_close\IF2003.csv', index_col = 0)
result = vasicek(0.3, a, b, r)
rate = result[1]
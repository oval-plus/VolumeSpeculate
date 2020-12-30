# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import json
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import cufflinks as cf
from statsmodels.tsa.arima.api import ARIMA
import statsmodels.api as sm
from cachetools import TTLCache, cached
from volume_look.util.utility import Utility
from volume_look.util.save_util import SaveUtil
from volume_look.analysis.build_data import BuildData
from volume_look.analysis.maluyao import Maluyao
cf.go_offline()
plotly.offline.init_notebook_mode(connected = True)

config_path = "/home/lky/volume_speculate/volume_look/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

util = Utility(config)

start_date = "20200103"
end_date = "20200301"
date_lst = util.generate_date_lst(start_date, end_date)
save_path = '/home/lky/volume_speculate/output/maluyao/spread/'

for i in range(0, len(date_lst)):
    date = date_lst[i]
    mly = Maluyao(True)
    spread = mly.cal_spread(util, date)
    spread.to_csv(save_path + date + '.csv', header = True)


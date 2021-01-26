ent\SJTU\thesis\program\df.csv', index_col = 0)
# k = 2
# ulist = []
# omegalist = []
# for i in range(40, len(datelist)):
#     i0 = i - 40
#     i1 = i0 + 10
#     i2 = i - 5
#     if datelist[i] == '20200102':
#         print('daole')
#     df['date'] = df['date'].apply(str)
#     df_0_1 = df[(df['date']>=datelist[i0]) & (df['date']<=datelist[i1])]
#     df_1_2 = df[(df.date>=datelist[i1])&(df.date<=datelist[i])]
#     df_0_2 = df[(df.date>=datelist[i0])&(df.date<=datelist[i])]
#     overnight_df_0_1 = df_0_1[df_0_1.is_overnight.astype(float)==1]
#     mu = np.mean(overnight_df_0_1['spread'])
#     sigma = np.std(overnight_df_0_1['spread'])
#     X = df_1_2[np.abs(df_1_2['spread'] - mu) < k*sigma]
#     X_mean = np.mean(X[(X['date']>=datelist[i2])&(X['date']<=datelist[i])]['spread'])
#     X['spread'] = X['spread'] - X_mean
#     X['last_spread'] = X['spread'].shift(1)
#     #MLE:bound constrained optimization
#     args1 = (0.01,0.9, 0.01,0.9)
#     # cons = con(args1)
#     X2 = (X['spread'].astype(float))
#     X1 = (X['last_spread'].astype(float))
#     def log_likelihood(params):
#         theta,omega = params
#         delta = 1 / 250 /241
#         omega2 = omega*np.sqrt((1 - np.exp(-2*theta*delta))/(2*theta))
#         N = 241 * 31-1
#         L = -1 * (-N / 2 * np.log(2 * np.pi) - N * np.log(omega2) - np.sum(1 / (2 *
#         omega2 ** 2)*(X2 - X1 * np.exp(-theta * delta)) ** 2))
#         return L
#     params = [0.00001, 0.5]
#     # theta_omega=optimize.minimize(log_likelihood,x0=params,
#     # method='SLSQP',constraints=cons)
#     theta_omega = optimize.minimize(log_likelihood, x0 = params)
#     theta = theta_omega.x[0]
#     omega = theta_omega.x[1]
#     gt = np.mean(X.iloc[:-1].spread)
#     gt_last = np.mean(X.iloc[1:].spread)
#     u = 1 / theta * (gt - gt_last) + gt
#     ulist.append([datelist[i],u])
#     omegalist.append([datelist[i],np.std(X.spread)])


# # future CSI short rate model (TAO XUAN)
# # meet Mon or Tue/Wed, told Samuel (set up database/ server) ask TAOXUAN
# # pair trading signal
# # web 

# # websocket (protocol)
# # 
# # 
# # Opensource database, not SQL (MongoDB / Redis / ...)
# # financial data 
# # 
# # full calibration time structure CSI index futures
# # model deviating too much (?)
# # camen
# u_df = pd.DataFrame(ulist)
# u_df.columns = ['date','u']
# omega_df = pd.DataFrame(omegalist)
# omega_df.columns = ['date', 'omega']
# df1 = pd.merge(df, u_df)
# df1 = pd.merge(df1, omega_df)
# df1['up'] = df1['u'] + k * df1['omega']
# df1['down'] = df1['u'] - k * df1['omega']

# df1.loc[df1.spread > df1.up,'holding_close'] = -1
# df1.loc[df1.spread > df1.up,'holding_far'] = 1
# df1.loc[df1.spread < df1.down,'holding_close'] = 1
# df1.loc[df1.spread < df1.down,'holding_far'] = -1
# df1['portfolio_ret'] = df1['holding_close'] * df1['expret1'] + \
#     df1['holding_far'] * df1['expret2']
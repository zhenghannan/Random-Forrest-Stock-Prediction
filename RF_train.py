import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.__version__, np.__version__, sklearn.__version__


# # 1. Data Clean

def read_data():
    data = pd.read_csv('finalproject_training_xy.csv')
    # narrow down samples to be within +/- 50% monthly return
    data = data[data['m_ret_next'].abs()<0.5]
    data.sort_values('date', inplace=True)
    many_nan_variables = data.isnull().sum()[data.isnull().sum() > 10000].index
    data = data[[i for i in data.columns if i not in many_nan_variables]]
    data = data[data['date']!='2018-08-13']
    data = data.loc[~data['m_ret_next'].isnull(), :]
    return data, many_nan_variables
data, many_nan_variables = read_data()

# # 2. Feature Engineering

def MSE(a,b):
    x = (a-b)**2
    return np.mean(x)
def ACC(x, y):
    return np.nansum(x==y) / len(x)

def cut(a, level=0.5):
    return np.maximum(np.minimum(a, level), -level)

def deal_with_financial_variables(data, select_variables=['actq'],standard='mkvaltq'):
    datacopy = data.copy()
    for i in select_variables:
        if i != standard:
            datacopy[i+'_'+standard] = datacopy[i] / datacopy[standard]
    return datacopy[[i+'_'+standard for i in select_variables if i !=standard]].fillna(0)

def deal_with_technical_variables(data):
    datacopy = data.copy()
    datacopy['spread_over_close'] = (datacopy['m_high_adj'] - datacopy['m_low_adj']) / datacopy['close_adj']
    datacopy['power_over_close'] = np.maximum(datacopy['m_high_adj']-datacopy['close_adj'],datacopy['close_adj']-datacopy['m_low_adj']) / datacopy['close_adj']
    datacopy['ln_ret'] = np.log(1+datacopy.m_ret)
    datacopy['Size'] = np.log(datacopy.mkvaltq)
    datacopy['BP'] = (datacopy['lseq']-datacopy['ltq'])/datacopy['cshprq']/datacopy['close_adj']
    datacopy['CETOP'] = datacopy['oancfy']/datacopy['mkvaltq']
    datacopy['ETOP'] = datacopy['revtq']/datacopy['mkvaltq']
    datacopy['MLEV'] = datacopy['lltq']/datacopy['mkvaltq']
    datacopy['DTOA'] = datacopy['atq']/datacopy['ltq']
    datacopy['STOM'] = datacopy['m_volume_adj']/datacopy['cshprq']
    return cut(datacopy[['Size','BP','ln_ret','CETOP','ETOP','MLEV','DTOA','STOM','spread_over_close','power_over_close','m_ret','m_volume_adj']].fillna(0), 20)

def deal_with_macro_variables(data):
    datacopy = data.copy()
    datacopy['SPYvolatility'] = (datacopy['SP500WeeklyHigh'] - datacopy['SP500WeeklyLow']) / datacopy['SP500WeeklyClose']
    return datacopy[['SPYvolatility','Bullish', 'Neutral', 'Bearish', 'Bullish8WeekMovAvg']]
        
def get_rolling_data(data, select_variables=['m_ret'],rolling_period=3):
    datacopy = data[select_variables].copy()
    df = data.copy()
    
    rollmean = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False).apply(lambda x:x.rolling(3).mean())
    rollmean = rollmean.reset_index()
    rollmax = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False).apply(lambda x:x.rolling(3).max())
    rollmax = rollmax.reset_index()
    rollmin = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False).apply(lambda x:x.rolling(3).min())
    rollmin = rollmin.reset_index()

    financial_variables_rmean = [c + '_rolling_mean_{}'.format(rolling_period) for c in rollmean.columns[2:]]
    rollmean.columns = ['date','comp_id'] + financial_variables_rmean
    financial_variables_rmin = [c + '_rolling_min_{}'.format(rolling_period) for c in rollmin.columns[2:]]
    rollmin.columns = ['date','comp_id'] + financial_variables_rmin
    financial_variables_rmax = [c + '_rolling_max_{}'.format(rolling_period) for c in rollmax.columns[2:]]
    rollmax.columns = ['date','comp_id'] + financial_variables_rmax

    datacopy = pd.merge(data[['date','comp_id']],rollmean,how='left',on=['date','comp_id'])
    datacopy = pd.merge(datacopy,rollmin,how='left',on=['date','comp_id'])
    datacopy = pd.merge(datacopy,rollmax,how='left',on=['date','comp_id'])
    return datacopy

y = 'm_ret_next'

financial_variables = ['actq', 'atq',
       'ceqq', 'cheq', 'chq', 'ciq', 'csh12q', 'cshfd12', 'cshfdq', 'cshopq',
       'cshprq', 'dd1q', 'dlttq', 'dpq', 'epsf12', 'epsfxq', 'epspxq',
       'epsx12', 'esopctq', 'gdwlq', 'ibadjq', 'ibcomq', 'ibmiiq', 'ibq',
       'intanq', 'invtq', 'lctq', 'lltq', 'loq', 'lseq', 'ltq', 'niq', 'nopiq',
       'oiadpq', 'oibdpq', 'piq', 'rdipq', 'rectq', 'req', 'revtq', 'tfvaq',
       'tfvceq', 'tfvlq', 'txdbclq', 'txtq', 'xaccq', 'xintq', 'xoprq', 'xrdq',
       'fincfy', 'intpny', 'ivncfy', 'oancfy', 'txpdy', 'dvpspq','dvpsxq', 'mkvaltq']
group_variables = ['ggroup', 'gind', 'gsector', 'gsubind', 'sic']
financial_variables = [i for i in financial_variables if i not in many_nan_variables]
technical_variables = ['close_adj','m_volume_adj', 'm_high_adj', 'm_low_adj', 'm_divs', 'm_ret']
macro_variables = ['Bullish', 'Neutral', 'Bearish', 'Bullish8WeekMovAvg','SP500WeeklyHigh', 'SP500WeeklyLow', 'SP500WeeklyClose']
continuous_factor = financial_variables + technical_variables

## data with lags
# financial var
data1 = deal_with_financial_variables(data, financial_variables,standard='mkvaltq').fillna(0)
data2 = deal_with_financial_variables(data, financial_variables,standard='atq')

# technical var
data3 = deal_with_technical_variables(data)

# macro var
data4 = deal_with_macro_variables(data)

# lag data
data_new = pd.concat([data[['date','comp_id']], data1, data2, data3, data4], axis=1)
need_to_lag_variables = data1.columns.tolist() + data2.columns.tolist()+ data3.columns.tolist() + data4.columns.tolist()

data5 = get_rolling_data(data_new, need_to_lag_variables, 3)
data5 = pd.merge(data_new, data5, on=['date','comp_id'], how='left')[data5.columns]
data5.index = data_new.index
data5.drop('date',inplace=True,axis=1)
data5.drop('comp_id',inplace=True,axis=1)

data6 = pd.get_dummies(data['gind'])

# # lag data
data_new = pd.concat([data[['date']], data1, data2, data3, data4, data5, data6, data[['m_ret_next']]],axis=1).dropna(subset=['m_ret_rolling_max_3']).fillna(0)
data_new.head()

data_new = data_new.sort_values('date')
data_train, data_val, data_test = data_new.iloc[:60000, :], data_new.iloc[60000:80000, :], data_new.iloc[80000::, :]
x_train, y_train = data_train.iloc[:, 1:-1],data_train.iloc[:, -1] 
x_val, y_val = data_val.iloc[:, 1:-1],data_val.iloc[:, -1] 
x_test, y_test = data_test.iloc[:, 1:-1],data_test.iloc[:, -1] 

x_train.shape


# # 2. model for predicting the returns

# ### baseline

baseline_train = MSE(y_train.values, np.zeros(y_train.shape[0]))
baseline_val = MSE(y_val.values, np.zeros(y_val.shape[0]))
baseline_test = MSE(y_test.values, np.zeros(y_test.shape[0]))
print(round(MSE(y_train.values, np.zeros(y_train.shape[0])), 5))
print(round(MSE(y_val.values, np.zeros(y_val.shape[0])),5))
print(round(MSE(y_test.values, np.zeros(y_test.shape[0])),5))


# ### simple model

model = LinearRegression(normalize=True, fit_intercept=True).fit(x_train, y_train)

train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar')


# # Lasso

model = Lasso(alpha=0.00001, normalize=True, fit_intercept=True,max_iter=10).fit(x_train, y_train)
print((model.coef_!=0).sum())
train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar')


# ### Lasso gridSearch
from  sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

k_range=[]
for i in range(100,100000,100):
    k_range.append(1/i)
max_iter_range=range(1,100)
#
param_gird = dict(alpha=k_range,max_iter=max_iter_range)

grid = RandomizedSearchCV(model,param_gird,cv=10,scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)


model = Lasso(alpha=grid.best_params_['alpha'], normalize=True, fit_intercept=True,max_iter=grid.best_params_['max_iter']).fit(x_train, y_train)
print((model.coef_!=0).sum())
train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar',title='Lasso')
plt.savefig('1.jpg')


# ### 3. Decision Trees

from sklearn import tree
model = tree.DecisionTreeRegressor(max_depth=50,min_samples_leaf=400).fit(x_train, y_train)

train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar',title='DecisionTree')
plt.savefig('3.jpg')


# ### Decision tree gridSearch

max_depth_range=range(1,50)
min_samples_leaf_range=range(100,500,10)
max_features_range=range(20,420,50)
#
param_gird = dict(max_depth=max_depth_range,min_samples_leaf=min_samples_leaf_range,max_features=max_features_range)

grid = RandomizedSearchCV(model,param_gird,cv=10,scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)


model = tree.DecisionTreeRegressor(max_depth=grid.best_params_['max_depth'],min_samples_leaf=grid.best_params_['min_samples_leaf'],max_features=grid.best_params_['max_features']).fit(x_train, y_train)

train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar')


# ## complicate model

# ### 3. Random Forest


from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=20,max_depth=25,min_samples_leaf=500).fit(x_train, y_train)

train, val, test = MSE(y_train.values, cut(model.predict(x_train))),MSE(y_val.values, cut(model.predict(x_val))),MSE(y_test.values, cut(model.predict(x_test)))
print(round(MSE(y_train.values, cut(model.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar',title='RandomForest')
plt.savefig('4.jpg')


# ### Random Forest gridSearch

max_depth_range=range(10,50)
min_samples_leaf_range=range(100,800,100)
max_features_range=range(10,200,10)
#
param_gird = dict(max_depth=max_depth_range,min_samples_leaf=min_samples_leaf_range,max_features=max_features_range)

grid = RandomizedSearchCV(model,param_gird,cv=10,scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)


pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(20).iloc[::-1].plot(kind='barh')
plt.savefig('7.jpg')


# ### 4. Gradient Boosting Tress

from sklearn import ensemble
model7 = ensemble.GradientBoostingRegressor(n_estimators=50,max_depth=25,min_samples_leaf=500).fit(x_train, y_train)

train, val, test = MSE(y_train.values, cut(model7.predict(x_train))),MSE(y_val.values, cut(model7.predict(x_val))),MSE(y_test.values, cut(model7.predict(x_test)))
print(round(MSE(y_train.values, cut(model7.predict(x_train))), 5))
print(round(MSE(y_val.values, cut(model7.predict(x_val))),5))
print(round(MSE(y_test.values, cut(model7.predict(x_test))),5))
pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar')


# ### save model

import pickle
with open('./mse_model.pkl', 'wb') as f:
    pickle.dump({'model':model7,'drop_var':many_nan_variables,'used_var':x_train.columns}, f)

pd.DataFrame([[baseline_train, baseline_val],[train, val]], index=['baseline','test'],columns=['train','val']).T.plot(kind='bar',title='GBDT')
plt.savefig("5.jpg")
print(round(MSE(y_test.values, cut(model7.predict(x_test))),5))


# ### result from predicting returns(decision tree)
model = model7
np.sum(np.sign(y_train.values) == np.sign(cut(model.predict(x_train)))) / len(y_train.values)
np.sum(np.sign(y_val.values) == np.sign(cut(model.predict(x_val)))) / len(y_val.values)
np.sum(np.sign(y_test.values) == np.sign(cut(model.predict(x_test)))) / len(y_test.values)

data_train, data_val, data_test = data_new.iloc[:60000, :], data_new.iloc[60000:80000, :], data_new.iloc[80000::, :]
x_train, y_train = data_train.iloc[:, 1:-1],np.sign(data_train.iloc[:, -1] )
x_val, y_val = data_val.iloc[:, 1:-1],np.sign(data_val.iloc[:, -1] )
x_test, y_test = data_test.iloc[:, 1:-1],np.sign(data_test.iloc[:, -1] )
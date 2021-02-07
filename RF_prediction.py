'''
approximate running time: 30s.

'''

import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.ensemble import AdaBoostRegressor
import warnings
import sys
warnings.filterwarnings('ignore')

print('''
The current version is
pandas {}
numpy {}
sklearn {}
python {}
    '''.format(pd.__version__, np.__version__, sklearn.__version__,  sys.version))


def MSE(a,b):
    x = (a-b)**2
    return np.mean(x)

def cut(a, level=0.5):
    return np.maximum(np.minimum(a, level), -level)

def next_year_mon(x):
    if x[-2:] == '12':
        return str(int(x[:4])+1)+'-01'
    else:
        mon = str(int(x[-2:])+1)
        return x[:4]+'-'+'0'*(2-len(mon))+mon

def read_data(many_nan_variables):
    data = pd.read_csv('finalproject-test.csv')
    data.sort_values('date', inplace=True)
    data = data[[i for i in data.columns if i not in many_nan_variables]]
    data = data[data['date']!='2018-08-13']
    return data


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
    
    rollmean = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False)\
            .apply(lambda x:x.rolling(3).mean())
    rollmean = rollmean.reset_index()
    rollmax = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False)\
            .apply(lambda x:x.rolling(3).max())
    rollmax = rollmax.reset_index()
    rollmin = df.set_index(['date','comp_id']).groupby(axis=0,level=[1], group_keys=False)\
            .apply(lambda x:x.rolling(3).min())
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

def create_prediction():
    with open('./mse_model.pkl','rb') as f:
        tmp = pickle.load(f)
        model = tmp['model']
        many_nan_variables = tmp['drop_var']
        used_variables = tmp['used_var']

    data = read_data(many_nan_variables)

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

    # group
    data6 = pd.get_dummies(data['gind'])

    # lag data
    data_new = pd.concat([data[['date','comp_id']], data1, data2, data3, data4, data5, data6],axis=1).dropna(subset=['m_ret_rolling_max_3']).fillna(0)
    data_new = data_new.sort_values('date')

    data_x = data_new.loc[:, used_variables]

    result = data_new.iloc[:, :2]
    result['m_ret'] = model.predict(data_x)
    result['year_mon'] = result['date'].apply(lambda x:x[:7])
    result['next_year_mon'] = result['year_mon'].apply(next_year_mon)
    result['year_pred'] = result['next_year_mon'].apply(lambda x:x[:4])
    result['month_pred'] = result['next_year_mon'].apply(lambda x:x[-2:])
    result = result[['year_pred','month_pred','comp_id','m_ret']]
    result.to_csv('returns.csv', index=False)

    datacopy = data[['date','comp_id','m_ret']]
    datacopy['year_pred'] = datacopy['date'].apply(lambda x:x[:4])
    datacopy['month_pred'] = datacopy['date'].apply(lambda x:x[5:7])
    datacopy['m_ret_result'] = datacopy['m_ret']
    datacopy = datacopy[['year_pred','month_pred','comp_id','m_ret_result']]
    final = pd.merge(result, datacopy, on=['year_pred','month_pred','comp_id'], how='inner')
    print(MSE(final['m_ret'],cut(final['m_ret_result'], 0.5)))

create_prediction()



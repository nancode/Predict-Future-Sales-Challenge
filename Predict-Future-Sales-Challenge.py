import numpy as np
import pandas as pd  
import lightgbm as lgb
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

path = './'

def pre_process_data():
    print('Preprocessing data..')
    s = pd.read_csv('./sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
    val = pd.read_csv('%s/test.csv' % path)
    # subset = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    median = s[(s.shop_id == 32) & (s.item_id == 2973) & (s.date_block_num == 4) & (
            s.item_price > 0)].item_price.median()
    s.loc[s.item_price < 0, 'item_price'] = median
    s['item_cnt_day'] = s['item_cnt_day'].clip(0, 1000)
    s['item_price'] = s['item_price'].clip(0, 300000)
    # Using values from https://www.kaggle.com/dlarionov/feature-engineering-xgb/notebook 
    s.loc[s.shop_id == 0, 'shop_id'] = 57
    val.loc[val.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    s.loc[s.shop_id == 1, 'shop_id'] = 58
    val.loc[val.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    s.loc[s.shop_id == 10, 'shop_id'] = 11
    val.loc[val.shop_id == 10, 'shop_id'] = 11
    # Rearranging for monthly sales
    df = s.groupby([s.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().clip(0, 20).reset_index()
    df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
    df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
    data = pd.merge(val, df, on=['item_id', 'shop_id'], how='left').fillna(0)
    data['item_id'] = np.log1p(data['item_id'])
    print('Preprocessing Done! ')
    return data

def linear_model(x_train, y_train):
    print('Training Linear Model..')
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    print('Predicting using Linear Model..')
    y_pre = linear_regression.predict(x_train)
    print('RMSE for linear model:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    print('MSE for linear model:', (mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    return linear_regression


def light_gbm_model(x_train, y_train):
    lgbm_parameters = {
        'feature_fraction': 1,
        'metric': 'rmse',
        'min_data_in_leaf': 16,
        'bagging_fraction': 0.85,
        'learning_rate': 0.03,
        'objective': 'mse',
        'bagging_seed': 2 ** 7,
        'num_leaves': 32,
        'bagging_freq': 3,
        'verbose': 0
    }
    print('Training Light gbm Model..')
    estimator = lgb.train(lgbm_parameters, lgb.Dataset(x_train, label=y_train), 300)
    print('Predicting using  Light gbm Model..')
    y_pre = estimator.predict(x_train)
    print('RMSE lgbm:', np.sqrt(mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))
    print('MSE lgbm:', (mean_squared_error(y_train.clip(0., 20.), y_pre.clip(0., 20.))))    
    return estimator


def pre_data(data_type, reg, x_test):
    if reg is None:
        reg = joblib.load('%s/%s_model_weight.model' % (path, data_type))
    y_pre = reg.predict(x_test)
    return y_pre


test = pre_process_data()
test_date_info = test.drop(labels=['ID'], axis=1)
y_train_normal = test_date_info['2015-10']
x_train_normal = test_date_info.drop(labels=['2015-10'], axis=1)
x_train_normal.columns = np.append(['shop_id', 'item_id'],
                                   np.arange(0, 33, 1))
linear_model = linear_model(x_train_normal, y_train_normal)
light_gbm_model = light_gbm_model(x_train_normal, y_train_normal)
test_x = test_date_info.drop(labels=['2013-01'], axis=1)
test_x.columns = np.append(['shop_id', 'item_id'],np.arange(0, 33, 1))
test_y_lgbm = pre_data('light_gbm', light_gbm_model, test_x)
test_y_linear = pre_data('linear', linear_model, test_x)

test['item_cnt_month'] = test_y_lgbm
test['item_cnt_month'] = test['item_cnt_month'].clip(0, 20)
test[['ID', 'item_cnt_month']].to_csv('lgbm.csv', index=False)

test['item_cnt_month'] = test_y_linear
test['item_cnt_month'] = test['item_cnt_month'].clip(0, 20)
test[['ID', 'item_cnt_month']].to_csv('linear.csv' , index=False)




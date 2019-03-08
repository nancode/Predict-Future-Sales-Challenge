import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

sales = pd.read_csv('./sales_train.csv')
item_cat = pd.read_csv('./item_categories.csv')
items = pd.read_csv('./items.csv')
shops = pd.read_csv('./shops.csv')
sample_submission = pd.read_csv('./sample_submission.csv')
test_data = pd.read_csv('./test.csv')

sales['date'] = pd.to_datetime(sales['date'],format = '%d.%m.%Y')
data = sales.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0)
data.reset_index(inplace = True)

data.head()
data = pd.merge(test_data,data,on = ['item_id','shop_id'],how = 'left')
data.fillna(0,inplace = True)
data.head()
Xtrain = np.expand_dims(data.values[:,:-1],axis = 2)
ytrain = data.values[:,-1:]
Xttest = np.expand_dims(data.values[:,1:],axis = 2)

model = Sequential()
model.add(LSTM(units = 64,input_shape = (36,1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error','accuracy'])
model.summary()
model.fit(Xtrain,ytrain,batch_size = 4096,epochs = 10)
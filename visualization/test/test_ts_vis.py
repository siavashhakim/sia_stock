from visualization import TsVisual
from data_access_layer import DataAccess
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

x = [1,2,4,5]
y = [30,20,40,60]
TsVisual.scatter_opt(y,x,'x','y','test_scatter_plot')


# test naive forecast
ticker_symbol = 'TSLA'
start_date = '2010-11-01'
end_date = '2021-01-01'
perd = '1mo'
ticker_df = DataAccess.data_auto_reader(ticker_symbol,start_date,end_date,perd)
tsla,date = DataAccess.data_cleaner(ticker_df)
train, test = DataAccess().ts_train_test_split(tsla)
dd= np.asarray(train)
y_hat = np.multiply(dd[len(dd)-1],np.ones(len(test)))
rms = sqrt(mean_squared_error(test, y_hat))
ts_name = ticker_symbol + '_daily'
date_cut = date[0:len(tsla)]
TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,date_cut,"Naive_Forecast")
from signal_processing import TsForecast
from data_access_layer import DataAccess
from datetime import datetime

ticker_symbol = 'TSLA'
end_traindate = '2021-01-01'
end_preddate = '2021-05-01'
start_traindateD = '2019-11-01'
end_realdata =  datetime.now().strftime("%Y") + '-' + \
                datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d") #'2021-02-20'


"""
/**********************************************************************
               Time-series Forecasting using daily data
***********************************************************************/
"""
perd = '1d'
ticker_df = DataAccess.data_auto_reader(ticker_symbol,start_traindateD,end_traindate,perd)
tsla,date = DataAccess.data_cleaner(ticker_df)
date_cut = date[0:len(tsla)]

ticker_df = DataAccess.data_auto_reader(ticker_symbol,end_traindate,end_realdata,perd)
tslam = DataAccess.data_publisher(ticker_df,perd)
TsForecast(date_cut,end_traindate,end_preddate,perd,ticker_symbol).time_series_forecasting_advanced()




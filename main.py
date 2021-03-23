# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 11:18:19 2019

@author: Siavash

TESLA Stock price forecasting

# Data source:
oil price:
https://datahub.io/core/oil-prices

Stock prices:
https://finance.yahoo.com/quote

Gold price:
https://datahub.io/core/gold-prices

"""
import os
import pandas as pd
import numpy as np
from helper_functions import Singnal_processing_travelTime, Data_sia_reader,Time_series_forecasting_simple
from helper_functions import Time_series_forecasting_advanced,Daily_Data_sia_reader,create_layout,layout_vis
from helper_functions import Data_sia_reader_auto,comp_info,Data_sia_reader_auto_index
import panel as pn
from datetime import datetime

main_dir = os.getcwd()

"""
/**********************************************************************
              Inputs 
***********************************************************************/
"""

tickerSymbol  = 'TSLA'

Start__traindateM = '2010-11-01'
End__traindateM = '2021-01-01'
Start__PreddateM = '01/01/2021'
End__PreddateM = '05/01/2021'

Start__traindateD = '2019-11-01'
End__traindateD = End__traindateM
Start__PreddateD = Start__PreddateM
End__PreddateD = End__PreddateM

End_realdata =  datetime.now().strftime("%Y") + '-' + \
                datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d") #'2021-02-20'


"""
/**********************************************************************
              Reading Data
***********************************************************************/
"""
df_merged = Data_sia_reader_auto(tickerSymbol,Start__traindateM,End__traindateM,'1mo')

df_merged = df_merged.fillna(method='bfill')
TSLA = df_merged.loc[:, 'Close']
DATE = df_merged.loc[:, 'Date']

"""
/**********************************************************************
              Missing data (NaN) -- Interpolation
***********************************************************************/
"""
TSLAm = TSLA.interpolate()


"""
/**********************************************************************
              Publishing cleaned data
***********************************************************************/
"""
df_mergedP = Data_sia_reader_auto(tickerSymbol,Start__traindateM,End_realdata,'1mo')
df_mergedP = df_mergedP.fillna(method='bfill')
TSLAP = df_mergedP.loc[:, 'Close']
DATEP = df_mergedP.loc[:, 'Date']
TSLAmP = TSLAP.interpolate()

d = {'Date': DATEP, 'Price': TSLAmP}
output = pd.DataFrame(data=d)
output.to_csv('cleaned_data_monthly.csv')

# Calcualting cross-correaltion between time-series & time delay
#[LandaOG,Time_pred_OG] = Singnal_processing_travelTime(Oil,Gold,'Oil','Gold',DATE)


"""
/**********************************************************************
              Changepoint detection
***********************************************************************/
"""


# Use stochastic methods to find the major change points


"""
/**********************************************************************
               Time-series Forecasting using monthly data
***********************************************************************/
"""
wn = 0
start_date = 0 #len(APPLE) - 3*12

a = TSLA[start_date:len(TSLA)]
ts_name = tickerSymbol
DATE_cut = DATE
index_cut = 0

nt = int(np.round(len(a)*.7))

#start_date = nt - 3*12

train=a[0:nt]
test=a[nt:]

df = pd.DataFrame(np.array(a), columns = list("q"))
train=df[0:nt]
test=df[nt:]


"""
/**********************************************************************
               Time-series Forecasting, simple methods
               
***********************************************************************/
"""
#Time_series_forecasting_simple(train,test,DATE_cut,index_cut,wn,ts_name,'M')

"""
/**********************************************************************
               Time-series Forecasting, advanced methods
               
***********************************************************************/
"""
# time consuming
PeriodTime_study = pd.date_range(start=Start__PreddateM, end=End__PreddateM, freq='MS') #After test period of time
PTime = pd.DataFrame(data=PeriodTime_study, columns = ['date'])
Time_series_forecasting_advanced(train,test,DATE_cut,index_cut,wn,ts_name,PTime,'M')


"""
/**********************************************************************
              Daily Data
***********************************************************************/
"""
df_merged = Data_sia_reader_auto(tickerSymbol,Start__traindateD,End__traindateD,'1d')

df_merged = df_merged.fillna(method='bfill')
df_merged = df_merged.fillna(method='ffill')
TSLA = df_merged.loc[:, 'Close']
DATE = df_merged.loc[:, 'Date']


"""
/**********************************************************************
              Missing data (NaN) -- Interpolation
***********************************************************************/
"""
TSLAm = TSLA.interpolate()


"""
/**********************************************************************
              Publishing cleaned data
***********************************************************************/
"""
df_mergedP = Data_sia_reader_auto(tickerSymbol,End__traindateM,End_realdata,'1d')

df_mergedP = df_mergedP.fillna(method='bfill')
df_mergedP = df_mergedP.fillna(method='ffill')
TSLAP = df_mergedP.loc[:, 'Close']
DATEP = df_mergedP.loc[:, 'Date']

TSLAmP = TSLAP.interpolate()
d = {'Date': DATEP, 'Price': TSLAmP}
output = pd.DataFrame(data=d)
output.to_csv('cleaned_data_daily.csv')

# Calcualting cross-correaltion between time-series & time delay
#[LandaOG,Time_pred_OG] = Singnal_processing_travelTime(Oil,Gold,'Oil','Gold',DATE)

"""
/**********************************************************************
              Changepoint detection
***********************************************************************/
"""


# Use stochastic methods to find the major change points


"""
/**********************************************************************
               Time-series Forecasting using daily data
***********************************************************************/
"""
wn = 0

start_date = 0 #len(AAPLEm) - 3*365

a = TSLAm[start_date:len(TSLAm)]
ts_name = tickerSymbol + '_daily'
DATE_cut = DATE[start_date:len(TSLAm)]
index_cut = 0

nt = int(np.round(len(a)*.7))


train=a[0:nt]
test=a[nt:]

df = pd.DataFrame(np.array(a), columns = list("q"))
df['q'] = df['q'].astype(float)
train=df[0:nt]
test=df[nt:]


"""
/**********************************************************************
               Time-series Forecasting, simple methods
               
***********************************************************************/
"""
#Time_series_forecasting_simple(train,test,DATE_cut,index_cut,wn,ts_name,'D')

"""
/**********************************************************************
               Time-series Forecasting, advanced methods
               
***********************************************************************/
"""
# time consuming
PeriodTime_study = pd.date_range(start=Start__PreddateD, end=End__PreddateD) #After test period of time
PTime = pd.DataFrame(data=PeriodTime_study, columns = ['date'])
Time_series_forecasting_advanced(train,test,DATE_cut,index_cut,wn,ts_name,PTime,'D')

"""
/**********************************************************************
               Financial analysis
***********************************************************************/
"""
app_col = create_layout(tickerSymbol)
pn.serve(app_col)




"""
/**********************************************************************
               Stock Index
***********************************************************************/
"""

Start__traindateD = '2019-01-01'
End__traindateD = End_realdata


aord = Data_sia_reader_auto_index('^AORD',Start__traindateD,End__traindateD,'1d')
nikkei = Data_sia_reader_auto_index('1360.T',Start__traindateD,End__traindateD,'1d')
hsi = Data_sia_reader_auto_index('^HSI',Start__traindateD,End__traindateD,'1d')
daxi = Data_sia_reader_auto_index('DAX',Start__traindateD,End__traindateD,'1d')
cac40 = Data_sia_reader_auto_index('^FCHI',Start__traindateD,End__traindateD,'1d')
sp500 = Data_sia_reader_auto_index('^GSPC',Start__traindateD,End__traindateD,'1d')
dji = Data_sia_reader_auto_index('^DJI',Start__traindateD,End__traindateD,'1d')
nasdaq = Data_sia_reader_auto_index('^IXIC',Start__traindateD,End__traindateD,'1d')
spy = Data_sia_reader_auto_index('SPY',Start__traindateD,End__traindateD,'1d')

tsla = Data_sia_reader_auto_index('TSLA',Start__traindateD,End__traindateD,'1d')
amc = Data_sia_reader_auto_index('AMC',Start__traindateD,End__traindateD,'1d')

# Due to the timezone issues, we extract and calculate appropriate stock market data for analysis
# Indicepanel is the DataFrame of our trading model
indicepanel=pd.DataFrame(index=spy.index)

indicepanel['spy']=spy['Open'].shift(-1)-spy['Open']
indicepanel['spy_lag1']=indicepanel['spy'].shift(1)
indicepanel['sp500']=sp500["Open"]-sp500['Open'].shift(1)
indicepanel['nasdaq']=nasdaq['Open']-nasdaq['Open'].shift(1)
indicepanel['dji']=dji['Open']-dji['Open'].shift(1)
indicepanel['tsla']=tsla['Open']-tsla['Open'].shift(1)
indicepanel['amc']=amc['Open']-amc['Open'].shift(1)

indicepanel['cac40']=cac40['Open']-cac40['Open'].shift(1)
indicepanel['daxi']=daxi['Open']-daxi['Open'].shift(1)

indicepanel['aord']=aord['Close']-aord['Open']
indicepanel['hsi']=hsi['Close']-hsi['Open']
indicepanel['nikkei']=nikkei['Close']-nikkei['Open']
indicepanel['Price']=spy['Open']


indicepanel = indicepanel.fillna(method='ffill')
indicepanel = indicepanel.dropna()


#split the data into (1)train set and (2)test set
Train = indicepanel.iloc[0:400, :]
Test = indicepanel.iloc[400:, :]
print(Train.shape, Test.shape)


# Generate scatter matrix among all stock markets (and the price of SPY) to observe the association
# from pandas.plotting import scatter_matrix
# sm = scatter_matrix(Train, figsize=(10, 10))

# Find the indice with largest correlation
corr_array = Train.iloc[:, :-1].corr()['nasdaq']
print(corr_array)

corr_array = Train.iloc[:, :-1].corr()['tsla']
print(corr_array)

corr_array = Train.iloc[:, :-1].corr()['amc']
print(corr_array)
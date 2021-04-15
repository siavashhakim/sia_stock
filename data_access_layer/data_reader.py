"""
Created on Thu May 31 16:14:32 2018

@author: Siavash Hakim Elahi

This package consists all the required function for reading stock data in real-time
"""

import pandas as pd
import yfinance as yf
import numpy as np

class DataAccess():

    def __init__(self):
        self.wn = 0
        self.st_date = 0
        self.index_cut = 0
        self.ts_ratio = .7

    @staticmethod
    def data_auto_reader(ticker_symbol,start_date,end_date,perd):
        #get data on this ticker
        ticker_data = yf.Ticker(ticker_symbol)
        #get the historical prices for this ticker
        ticker_df = ticker_data.history(interval=perd, start=start_date, end=end_date)
        # ticker_df = ticker_df.drop(['Open', 'High', 'Low','Volume','Dividends', 'Stock Splits'], axis=1)
        ticker_df = ticker_df.drop(['High', 'Low','Volume','Dividends', 'Stock Splits'], axis=1)
        ticker_df = ticker_df.reset_index()
        return ticker_df

    @staticmethod
    def data_cleaner(ts_df):
        ts_df = ts_df.fillna(method='bfill')
        ts_df = ts_df.fillna(method='ffill')
        stock = ts_df.loc[:, 'Close']
        stockm = stock.interpolate()
        date = ts_df.loc[:, 'Date']
        return stockm,date

    @staticmethod
    def data_publisher(ts_df,perd):
        ts_df = ts_df.fillna(method='bfill')
        ts_df = ts_df.fillna(method='ffill')
        stock = ts_df.loc[:, 'Close']
        date = ts_df.loc[:, 'Date']
        stockm = stock.interpolate()
        d = {'Date': date, 'Price': stockm}
        output = pd.DataFrame(data=d)
        if perd == '1mo':
            output.to_csv('cleaned_data_monthly.csv')
        else:
            output.to_csv('cleaned_data_daily.csv')
        return stockm


    def ts_train_test_split(self,stock):
        a = stock[self.st_date:len(stock)]
        nt = int(np.round(len(a)*self.ts_ratio))
        df = pd.DataFrame(np.array(a), columns = list("q"))
        train=df[0:nt]
        test=df[nt:]
        return train, test





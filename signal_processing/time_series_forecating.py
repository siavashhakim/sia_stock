"""
Created on Thu May 31 16:14:32 2018

@author: hakimels

This package consists all the required function for time series forecasting
"""

from sklearn.preprocessing import normalize
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import panel as pn
import holoviews as hv
import panel.interact as interact
from datetime import datetime
import yfinance as yf
from sklearn import preprocessing
from bokeh.models import LinearAxis
from bokeh.models import Range1d
from bokeh.models.renderers import GlyphRenderer
from visualization import TsVisual

class TsForecast():

    def __init__(self):
        pass



    def Time_series_forecasting_simple(self, train,test,DATE_cut,index_cut,wn,ts_name,falgDM):
        """
        /**********************************************************************
                       Method 1: Start with a Naive Approach

           We can infer from the graph that the price of the coin is stable from the start.
           Many a times we are provided with a dataset, which is stable throughout it’s time period.
           If we want to forecast the price for the next day, we can simply take the last day value and
           estimate the same value for the next day. Such forecasting technique which assumes that
           the next expected point is equal to the last observed point is called Naive Method.

           We can infer from the RMSE value and the graph above,
           that Naive method isn’t suited for datasets with high variability.
        ***********************************************************************/
        """
        dd= np.asarray(train)
        y_hat = np.multiply(dd[len(dd)-1],np.ones(len(test)))
        rms = sqrt(mean_squared_error(test, y_hat))
        print("Naive_Forecast RMSE:", str(rms))
        TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,DATE_cut,"Naive_Forecast")

        """
        /**********************************************************************
                       Method 2: Simple Average
                       
        ***********************************************************************/
        """
        y_hat_avg = np.multiply(float(train.mean()),np.ones(len(test)))
        rms = sqrt(mean_squared_error(test, y_hat_avg))
        print("Simple_Average RMSE:", str(rms))
        TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,DATE_cut,"Simple_Average")

        """
        /**********************************************************************
                       Method 3 – Moving Average
                       we will take the average of the prices for last few time periods only
                       
        ***********************************************************************/
        """

        # Hyper parameter tuning
        if falgDM =='M':
            train_rolling_list = np.arange(5, .9*len(train), 20)
        else:
            train_rolling_list = np.arange(5, .9*len(train), 100)
        rms_temp = np.zeros(len(train_rolling_list))

        for i in range(0,len(train_rolling_list)):
            print("iter #"+ str(i) +" out of "+ str(len(train_rolling_list)))
            y_hat_Movavg = np.multiply(float(train.rolling(int(train_rolling_list[i])).mean().iloc[-1]),np.ones(len(test)))
            rms_temp[i] = sqrt(mean_squared_error(test, y_hat_Movavg))


        TsVisual().scatter_opt(train_rolling_list,rms_temp,"seasonal periods","RMSE","Opt_seasonal_periods")

        rms_templist = list(rms_temp)
        optmimum_index = rms_templist.index(np.min(rms_templist))
        opt_rolling = train_rolling_list[optmimum_index]


        y_hat_Movavg = np.multiply(float(train.rolling(int(opt_rolling)).mean().iloc[-1]),np.ones(len(test)))
        rms = sqrt(mean_squared_error(test, y_hat_Movavg))
        print("Moving_Averag RMSE:", str(rms))
        TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,DATE_cut,"Moving_Average")

        """
        /**********************************************************************
                       Method 4 – Simple Exponential Smoothing
                       
                       After we have understood the above methods, we can note that 
                       both Simple average and Weighted moving average lie on completely opposite ends. 
                       We would need something between these two extremes approaches which takes 
                       into account all the data while weighing the data points differently. 
                       For example it may be sensible to attach larger weights to more 
                       recent observations than to observations from the distant past. 
                       The technique which works on this principle is called Simple exponential smoothing. 
                       Forecasts are calculated using weighted averages where the weights decrease 
                       exponentially as observations come from further in the past, 
                       the smallest weights are associated with the oldest observations:
                       
        ***********************************************************************/
        """

        # Hyper parameter tuning
        smoothing_level_list = np.arange(.1, 5, 1)
        rms_temp = np.zeros(len(smoothing_level_list))

        for i in range(0,len(smoothing_level_list)):
            print("iter #"+ str(i) +" out of "+ str(len(smoothing_level_list)))
            fit1 = SimpleExpSmoothing(np.asarray(train)).fit(smoothing_level=smoothing_level_list[i],optimized=False)
            y_hat_avg = fit1.forecast(len(test))
            rms_temp[i] = sqrt(mean_squared_error(test, y_hat_avg))

        TsVisual().scatter_opt(smoothing_level_list,rms_temp,"seasonal periods","RMSE","Opt_seasonal_periods")

        rms_templist = list(rms_temp)
        optmimum_index = rms_templist.index(np.min(rms_templist))
        opt_smoothing = smoothing_level_list[optmimum_index]

        fit2 = SimpleExpSmoothing(np.asarray(train)).fit(smoothing_level=opt_smoothing,optimized=False)
        y_hat_avgSES = fit2.forecast(len(test))
        rms = sqrt(mean_squared_error(test, y_hat_avgSES))
        print("Exponential_smoothing RMSE:", str(rms))
        TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,DATE_cut,"Exponential_smoothing")


        """
        /**********************************************************************
                       Method 5 – Holt’s Linear Trend method
                       
                       We have now learnt several methods to forecast but 
                       we can see that these models don’t work well on data with high variations. 
                       Consider that the price of the bitcoin is increasing.
                       
                       But we need a method that can map the trend accurately without 
                       any assumptions. Such a method that takes into account the 
                       trend of the dataset is called Holt’s Linear Trend method.
                       
                       
                       Each Time series dataset can be decomposed into it’s componenets 
                       which are Trend, Seasonality and Residual. Any dataset that follows a 
                       trend can use Holt’s linear trend method for forecasting
                       
                       
                       Holt extended simple exponential smoothing to allow forecasting of 
                       data with a trend. It is nothing more than exponential 
                       smoothing applied to both level(the average value in the series) 
                       and trend.
                       
        ***********************************************************************/
        """

        #sm.tsa.seasonal_decompose(train.q).plot()
        result = sm.tsa.stattools.adfuller(train.q)
        plt.show()

        # Hyper parameter tuning
        smoothing_level_list = np.arange(.1, 5, 1)
        rms_temp = np.zeros(len(smoothing_level_list))

        for i in range(0,len(smoothing_level_list)):
            print("iter #"+ str(i) +" out of "+ str(len(smoothing_level_list)))
            fit1 = Holt(np.asarray(train)).fit(smoothing_level = smoothing_level_list[i],smoothing_slope = 0.1)
            y_hat_avg = fit1.forecast(len(test))
            rms_temp[i] = sqrt(mean_squared_error(test, y_hat_avg))

        TsVisual().scatter_opt(smoothing_level_list,rms_temp,"seasonal periods","RMSE","Opt_seasonal_periods")

        rms_templist = list(rms_temp)
        optmimum_index = rms_templist.index(np.min(rms_templist))
        opt_smoothing = smoothing_level_list[optmimum_index]

        fit1 = Holt(np.asarray(train)).fit(smoothing_level = opt_smoothing,smoothing_slope = 0.1)
        y_hat_avgHolt = fit1.forecast(len(test))
        rms = sqrt(mean_squared_error(test, y_hat_avgHolt))
        print("Holt_Linear RMSE:", str(rms))
        TsVisual().ts_train_test(train,test,ts_name,rms,y_hat,DATE_cut,"Holt_linear_Trend")



    def Time_series_forecasting_advanced(self,train,test,DATE_cut,index_cut,wn,ts_name,PTime,falgDM):
        """
        /**********************************************************************
                      Method 6 – Holt-Winters Method
                      Seasonality

                      Hence we need a method that takes into
                      account both trend and seasonality to forecast future prices

                      The idea behind triple exponential smoothing(Holt’s Winter) is
                      to apply exponential
                      smoothing to the seasonal components in addition to level and trend


        Parameters:
        endog (array-like) – Time series
        trend ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of trend component.
        damped (bool, optional) – Should the trend component be damped.
        seasonal ({"add", "mul", "additive", "multiplicative", None}, optional) – Type of seasonal component.
        seasonal_periods (int, optional) – The number of seasons to consider for the holt winters.
        Returns:
        results

        Return type:
        ExponentialSmoothing class

        ***********************************************************************/
        """
        # Proposed seasional period by event detection algorithm
        #season_per = np.max(Events_diff)

        # Hyper parameter tuning
        if falgDM =='M':
            season_list = np.arange(5, int(len(train)*.9), 10)
        else:
            #season_list = np.arange(1500, int(len(train)*.9), 100)
            season_list = np.arange(int(len(train)*.3), int(len(train)*.9), 10)
        rms_temp = np.zeros(len(season_list))

        for i in range(0,len(season_list)):
            print("iter #"+ str(i) +" out of "+ str(len(season_list)))
            fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=season_list[i] , seasonal='mul').fit()
            y_hat_avgHolt_winter = fit1.forecast(len(test))
            #y_hat_avgHolt_winter_train = fit1.fittedvalues
            #rms_temp[i] = sqrt(mean_squared_error(train, y_hat_avgHolt_winter_train))
            rms_temp[i] = sqrt(mean_squared_error(test, y_hat_avgHolt_winter))


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.plot(season_list,rms_temp,'o')
        ax.grid()
        ax.set_ylabel("RMSE")
        ax.set_xlabel("seasonal periods")
        plt.savefig("Opt_seasonal_periods.png")
        plt.show()

        rms_templist = list(rms_temp)
        optmimum_index = rms_templist.index(np.min(rms_templist))
        opt_seasonPeriod = season_list[optmimum_index]


        y_hat_avg = test.copy()
        #fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=1000 ,trend='add', seasonal='add',).fit()
        #fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=1500 , trend = 'add', damped = True, seasonal='mul').fit()
        fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=opt_seasonPeriod, seasonal='mul').fit()
        y_hat_avgHolt_winter = fit1.forecast(len(test))

        #y_hat_avgHolt_winter2 = fit1.forecast(len(test)+int(.5*365))
        y_hat_avgHolt_winter2 = fit1.forecast(len(test)+len(PTime))

        #plt.plot(y_hat_avgHolt_winter2)

        #plt.plot(y_hat_avgHolt_winter)

        rms = sqrt(mean_squared_error(test, y_hat_avgHolt_winter))
        print(rms)



        PriceX = y_hat_avgHolt_winter2[len(test):len(test)+len(PTime)]
        d = {'Date': PTime['date'], 'Price': PriceX}
        output = pd.DataFrame(data=d)


        output.to_csv(ts_name+'_price_pred.csv')

        #Sensitivity analysis o seasonal period
        plt.figure(figsize=(12,8))
        plt.plot(DATE_cut[train.index + index_cut],train, label='Train')
        plt.plot(DATE_cut[test.index+ index_cut],test, label='Test')
        plt.plot(DATE_cut[test.index+ index_cut],y_hat_avgHolt_winter, label='Forecast')
        #plt.plot(PTime['date'],y_hat_avgHolt_winter2[len(test):len(test)+len(PTime)], label='ForecastExt')
        plt.legend(loc='best')
        plt.title("" + " and RMSE= "+str(rms))
        plt.savefig("Holt_winter"+ ts_name+str(wn)+".png")
        plt.show()



        """
        /**********************************************************************
                      Method 7 – ARIMA
                      Autoregressive Integrated Moving average
                      
                      While exponential smoothing models were based on a description of trend 
                      and seasonality in the data, ARIMA models aim to describe 
                      the correlations in the data with each other. 
                      An improvement over ARIMA is Seasonal ARIMA. 
                      It takes into account the seasonality of dataset just like Holt’ Winter method. 
                      You can study more about ARIMA and Seasonal ARIMA models and 
                      it’s pre-processing from these articles (1) and (2).
                       
        ***********************************************************************/
        """

        y_hat_avg = test.copy()
        #fit1 = sm.tsa.statespace.SARIMAX(train, order=(1, 0, 0), seasonal_order=(0,1,1,12),trend=(1,0,-1,0)).fit()
        fit1 = sm.tsa.statespace.SARIMAX(train, order=(4, 0, 0), seasonal_order=(1,1,1,7)).fit()
        y_hat_avgARIMA = fit1.predict(start=test.index[0], end=test.index[len(test)-1], dynamic=True)

        rms = sqrt(mean_squared_error(test, y_hat_avgARIMA))
        print(rms)

        plt.figure(figsize=(12,8))
        plt.plot(DATE_cut[train.index + index_cut],train, label='Train')
        plt.plot(DATE_cut[test.index+ index_cut],test, label='Test')
        plt.plot(DATE_cut[test.index+ index_cut],y_hat_avgARIMA, label='ARIMA')
        plt.legend(loc='best')
        plt.title("ARIMA" + "and RMSE= "+str(rms))
        plt.savefig("ARIMA"+ts_name+str(wn)+".png")
        plt.show()
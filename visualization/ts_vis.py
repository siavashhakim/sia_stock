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

class TsVisual():
    def __init__(self):
      self.index_cut = 0

    def ts_train_test(self,train,test,ts_name,rms,y_hat,date_cut,file_name):
        plt.figure(figsize=(12,8))
        plt.plot(date_cut[train.index + self.index_cut],train, label='Train')
        plt.plot(date_cut[test.index+ self.index_cut],test, label='Test')
        plt.plot(date_cut[test.index+ self.index_cut],y_hat, label=file_name)
        plt.legend(loc='best')
        plt.title(file_name + "and RMSE= "+str(rms))
        plt.savefig(file_name + "_" +ts_name+".png")

    @staticmethod
    def scatter_opt(y,x,xl,yl,file_name):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.plot(y,x,'o')
        ax.grid()
        ax.set_ylabel(yl)
        ax.set_xlabel(xl)
        plt.savefig(file_name + ".png")

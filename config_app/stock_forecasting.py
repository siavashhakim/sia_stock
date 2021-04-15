import panel as pn
from lambda_stock_ui import Page
import numpy as np
import holoviews as hv
from panel.interact import interact
import pandas as pd
from bokeh.models import LinearAxis
from bokeh.models import Range1d
from bokeh.models.renderers import GlyphRenderer
from sklearn import preprocessing
import yfinance as yf
from datetime import datetime,timedelta
import holidays
import calendar
from signal_processing import TsForecast
from data_access_layer import DataAccess


class StockForecasting(Page):

    def __init__(self, page_num=0, page_name='default', filename=''):
        """
        Args:
            page_num:
            page_name:
            filename:
        """
        super().__init__(page_num, page_name, filename)
        self.df_widget_idx = 2
        self.status = '### Ready'


    def create_layout(self):
        pn.extension()
        title = pn.pane.Markdown("""
                # Stock Market Forecasting tool (version alpha)
                """,background='#007500')

        title_recom = pn.pane.Markdown("""
                # Recommendations by other firms in the past 60 days
                """,background='#007500')

        title.session_id = self.session_id
        title_recom.session_id = self.session_id

        status_console = pn.pane.Markdown(self.status)

        # visualize table which contains company info
        comp_data = interact(self.comp_info, stock='TSLA')
        comp_data.session_id = self.session_id
        # df_info = comp_data[1][0].object
        # df_widget = pn.widgets.DataFrame(
        #     df_info, name='DataFrame', autosize_mode='fit_columns',width=400)
        # df_widget.session_id = self.session_id

        # visualize table which contains stock earnings
        # stk_news = interact(self.stock_news, stock='TSLA')
        # stk_news.session_id = self.session_id
        # df_info2 = stk_news[1][0].object
        # df_widget2 = pn.widgets.DataFrame(
        #     df_info2, name='DataFrame', autosize_mode='fit_columns',width=400)
        # df_widget2.session_id = self.session_id

        # visualize table which contains recommendations by other banks and traders
        stk_recom = interact(self.stock_recom, stock='TSLA')
        stk_recom.session_id = self.session_id
        # df_info3 = stk_recom[1][0].object
        # df_widget3 = pn.widgets.DataFrame(
        #     df_info3, name='DataFrame', autosize_mode='fit_columns',width=900,background='#00ff00')
        # df_widget3.session_id = self.session_id

        # Plot forecast and real data
        layout = interact(self.layout_vis, stock='TSLA')
        layout.session_id = self.session_id


        # app_col1 = pn.Column(layout[1],title_recom,df_widget3,width=2100)
        # app_col = pn.Column(title,layout[0],df_widget, df_widget2,width=400)
        # app_row = pn.Row(app_col,app_col1,margin=(25, 5, 25, 25),width=2100) #, background='#007500')
        app_col1 = pn.Column(title_recom,stk_recom[1][0],width=900)
        # app_col = pn.Column(title,layout[0],stk_recom[0],comp_data[0],stk_news[0],comp_data[1][0], stk_news[1][0],width=400)
        app_col = pn.Column(title,layout[0],stk_recom[0],comp_data[0],comp_data[1][0],width=400)
        app_row = pn.Row(app_col,layout[1],app_col1,margin=(25, 5, 25, 25),width=900) #, background='#007500')

        return app_row

    @staticmethod
    def forecast_model(stock):
        # Todo: make it interactive by using forecasting functions here
        perd = '1d'
        end_traindate = '2021-03-01'
        end_preddate = '2021-05-01'
        start_traindateD = '2019-11-01'
        end_realdata =  datetime.now().strftime("%Y") + '-' + \
                        datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d")

        ticker_df = DataAccess.data_auto_reader(stock,start_traindateD,end_traindate,perd)
        tsla,date = DataAccess.data_cleaner(ticker_df)
        date_cut = date[0:len(tsla)]

        ticker_df = DataAccess.data_auto_reader(stock,end_traindate,end_realdata,perd)
        tslam = DataAccess.data_publisher(ticker_df,perd)
        TsForecast(date_cut,end_traindate,end_preddate,perd,stock).time_series_forecasting_advanced()

    @staticmethod
    def layout_vis(stock):

        StockForecasting.forecast_model(stock)

        # comp_data = StockForecasting.comp_info(stock)
        # Todo: replace main_dir with self.main_dir using OOP --> replace all with S3 bucket
        main_dir = 'C:\\Users\\rob\\PycharmProjects\\sia_stock\\config_app'

        new_path = main_dir + str('\\' + stock + '_daily_price_pred.csv')
        data_pred = pd.read_csv(new_path)

        new_path_clean = main_dir + str('\\cleaned_data_daily.csv')
        data_real = pd.read_csv(new_path_clean)

        data_pred = StockForecasting.date_matching(data_pred)

        # moving average (smoothing)
        # data_real['MA10'] = data_real['Price'].rolling(10).mean()
        data_real['MA5'] = data_real['Price'].rolling(5).mean()
        data_real['MA10'] = data_real['Price'].rolling(10).mean()

        data_pred['MA5'] = data_pred['Price'].rolling(5).mean()
        data_pred['MA10'] = data_pred['Price'].rolling(10).mean()



        # L2 and L1 normalization
        # data_pred['Price'] = preprocessing.normalize(np.array([data_pred.Price.tolist()]), norm='l1')[0].tolist()
        # data_real['Price'] = preprocessing.normalize(np.array([data_real.Price.tolist()]), norm='l1')[0].tolist()

        # customized normalization
        # ratio0 = data_real.Price.tolist()[0] / data_pred.Price.tolist()[0]
        # data_pred['Price'] = np.multiply(data_pred.Price.tolist(), ratio0)

        hv_plt_pred = hv.Curve((data_pred.Date.tolist(), data_pred.Price.tolist()), label='Forecast').opts(
            height=400, width=1100,xrotation= 90,line_color='g',xlabel='Date',ylabel='Norm Price $',hooks=[StockForecasting.apply_formatter])

        hv_plt_pred_MA10 = hv.Curve((data_pred.Date.tolist(), data_pred.MA10.tolist()), label='Forecast MA10').opts(
            responsive=True, height=600, width=1200, xrotation= 90,line_color='yellow',xlabel='Date',ylabel='Norm Price $')


        hv_plt_pred_MA5 = hv.Curve((data_pred.Date.tolist(), data_pred.MA5.tolist()), label='Forecast MA5').opts(
            height=400, width=1100,xrotation= 90,line_color='orange',xlabel='Date',ylabel='Norm Price $',hooks=[StockForecasting.apply_formatter])


        hv_plt_real = hv.Curve((data_real.Date.tolist(), data_real.Price.tolist()), label='Real Data').opts(
            height=400, width=1100,xrotation= 90,line_color='r',xlabel='Date',ylabel='Norm Price $',hooks=[StockForecasting.apply_formatter])

        hv_plt_real_MA10 = hv.Curve((data_real.Date.tolist(), data_real.MA10.tolist()), label='Real Data MA10').opts(
            responsive=True, height=400, width=1100, xrotation= 90,line_color='b',xlabel='Date',ylabel='Norm Price $',hooks=[StockForecasting.apply_formatter])

        hv_plt_real_MA5 = hv.Curve((data_real.Date.tolist(), data_real.MA5.tolist()), label='Real Data MA5').opts(
            responsive=True, height=400, width=1100, xrotation= 90,line_color='cyan',xlabel='Date',ylabel='Norm Price $',hooks=[StockForecasting.apply_formatter])


        layout = hv.Layout(hv_plt_pred * hv_plt_real * hv_plt_pred_MA5)
        # layout = hv.Layout(hv_plt_pred+hv_plt_pred)

        return layout


    @staticmethod
    def apply_formatter(plot):
        p = plot.state

        # create secondary range and axis
        p.extra_y_ranges = {"twiny": Range1d(start=0, end=1)}
        p.add_layout(LinearAxis(y_range_name="twiny"), 'right')

        # set glyph y_range_name to the one we've just created
        glyph = p.select(dict(type=GlyphRenderer))[0]
        glyph.y_range_name = 'twiny'

        # set proper range
        glyph = p.renderers[-1]
        vals = glyph.data_source.data['y'] # ugly hardcoded solution, see notes below
        p.extra_y_ranges["twiny"].start = vals.min()* 0.99
        p.extra_y_ranges["twiny"].end = vals.max()* 1.01


    @staticmethod
    def comp_info(stock):
        #info on the company
        tickerData = yf.Ticker(stock)
        info_com = tickerData.info
        info_comDF  = pd.DataFrame.from_dict(info_com, orient='index')
        info_comDF = info_comDF.loc[['market','sector','open','regularMarketOpen','payoutRatio'
            ,'yield','ytdReturn','marketCap'
            ,'dividendRate','shortRatio','fiftyDayAverage','twoHundredDayAverage']]
        info_comDF2 = info_comDF.rename(columns={0: "info"})
        return info_comDF2

    @staticmethod
    def stock_news(stock):
        #info on the company
        tickerData = yf.Ticker(stock)
        st_news = tickerData.calendar
        if len(np.shape(st_news)) !=0:
            st_news2 = st_news.rename(columns={0: str(st_news.loc['Earnings Date'][0])})
            st_news2 = st_news2.rename(columns={1: str(st_news.loc['Earnings Date'][1])})
            st_news2 = st_news2.drop(['Earnings Date'])
        else:
            st_news2 = pd.DataFrame()
        return st_news2


    @staticmethod
    def stock_recom(stock):
        #info on the company
        tickerData = yf.Ticker(stock)
        st_recom = tickerData.recommendations
        if len(np.shape(st_recom)) !=0:
            st_recom2 = st_recom.reset_index()
            start_date = (datetime.now()-timedelta(days=120)).strftime("%d/%m/%Y %H:%M:%S")
            st_recom3 = st_recom2[(st_recom2['Date'] > start_date)]
            st_recom3 = st_recom3.reset_index().drop(columns=['index'])
        else:
            st_recom3 = pd.DataFrame()
        return st_recom3

    def _write_widget_df(self, df):
        """
        Args:
            df:
        """
        self.app_ref[self.page_num][self.df_widget_idx-1].value = df
        self.app_ref[self.page_num][self.df_widget_idx-1].height = 400


    def _write_status(self, new_status):
        print('write status')
        self.app_ref[self.page_num][self.df_widget_idx-2][2][4].object = new_status



    @staticmethod
    def workdays(int_d, end_d, excluded=(6, 7)):
        d = datetime.strptime(int_d, '%Y-%m-%d')
        end = datetime.strptime(end_d, '%Y-%m-%d')
        days = []
        while d.date() <= end.date():
            if d.isoweekday() not in excluded:
                days.append(d.strftime('%Y-%m-%d'))
            d += timedelta(days=1)
        return days

    @staticmethod
    def date_matching(data_pred):
        # removing holidays from the forecast
        us_holidays = holidays.US()
        drop_dates = us_holidays[np.min(data_pred.Date): np.max(data_pred.Date)]
        drop_dates2 = []
        for i in range(0,len(drop_dates)):
            drop_dates2.append(drop_dates[i].strftime('%Y-%m-%d'))

        # getting list of working/business day
        date_list = StockForecasting.workdays(np.min(data_pred.Date),np.max(data_pred.Date))
        business_day = pd.DataFrame(np.array(date_list),columns=['Date'])
        data_pred1 = pd.merge(business_day,data_pred, on ='Date')
        data_pred1 = data_pred1[['Date','Price']]

        ind_drop = []
        for i in range(0,len(drop_dates2)):
            ind_drop.append(data_pred1[data_pred1.Date ==  drop_dates2[i]].index[0])

        data_pred1 = data_pred1.drop(ind_drop)
        return data_pred1
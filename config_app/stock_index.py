import panel as pn
from lambda_stock_ui import Page
from panel.interact import interact
import pandas as pd
from datetime import datetime,timedelta
from data_access_layer import DataAccess
import holoviews as hv
from bokeh.models import LinearAxis
from bokeh.models import Range1d
from bokeh.models.renderers import GlyphRenderer


class StockIndex(Page):

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
                # Stock Index Correlation
                """,background='#007500')

        title.session_id = self.session_id

        output = interact(self.stockindex_model, stock='TSLA')
        output.session_id = self.session_id
        # df_info = output[1][0].object
        # df_widget = pn.widgets.DataFrame(
        #     df_info, name='DataFrame', autosize_mode='fit_columns',width=900)
        # df_widget.session_id = self.session_id

        # Plot forecast and real data
        layout = interact(self.layout_vis, stock='TSLA')
        layout.session_id = self.session_id

        layout2 = interact(self.layout_vis2, stock='TSLA')
        layout2.session_id = self.session_id

        app_col = pn.Column(title, output[0], output[1][0],layout[1],width=900)
        app_row = pn.Row(app_col,layout2[1],margin=(25, 5, 25, 25),width=900) #, background='#007500')

        return app_row


    @staticmethod
    def stockindex_model(stock):
        perd = '1d'
        end_traindate = '2021-01-01'
        start_traindateD = '2020-01-01'
        end_realdata =  datetime.now().strftime("%Y") + '-' + \
                        datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d")


        ticker_df = DataAccess.data_auto_reader(stock,start_traindateD,end_realdata,perd)
        aord = DataAccess.data_auto_reader('^AORD',start_traindateD,end_realdata,perd)
        nikkei = DataAccess.data_auto_reader('1360.T',start_traindateD,end_realdata,perd)
        hsi = DataAccess.data_auto_reader('^HSI',start_traindateD,end_realdata,perd)
        daxi = DataAccess.data_auto_reader('DAX',start_traindateD,end_realdata,perd)
        cac40 = DataAccess.data_auto_reader('^FCHI',start_traindateD,end_realdata,perd)
        sp500 = DataAccess.data_auto_reader('^GSPC',start_traindateD,end_realdata,perd)
        dji = DataAccess.data_auto_reader('^DJI',start_traindateD,end_realdata,perd)
        nasdaq = DataAccess.data_auto_reader('^IXIC',start_traindateD,end_realdata,perd)
        spy = DataAccess.data_auto_reader('SPY',start_traindateD,end_realdata,perd)


        # Due to the timezone issues, we extract and calculate appropriate stock market data for analysis
        # Indicepanel is the DataFrame of our trading model
        indicepanel=pd.DataFrame(index=spy.index)

        # US market
        indicepanel['spy'] = spy['Open'].shift(-1) - spy['Open']
        indicepanel['spy_lag1'] = indicepanel['spy'].shift(1)
        indicepanel['sp500'] = sp500["Open"] - sp500['Open'].shift(1)
        indicepanel['nasdaq'] = nasdaq['Open'] - nasdaq['Open'].shift(1)
        indicepanel['dji'] = dji['Open'] - dji['Open'].shift(1)
        indicepanel['target_stock'] = ticker_df['Open'] - ticker_df['Open'].shift(1)
        # EU market
        indicepanel['cac40'] = cac40['Open'] - cac40['Open'].shift(1)
        indicepanel['daxi'] = daxi['Open'] - daxi['Open'].shift(1)
        # Asian market
        indicepanel['aord'] = aord['Close'] - aord['Open']
        indicepanel['hsi'] = hsi['Close'] - hsi['Open']
        indicepanel['nikkei'] = nikkei['Close'] - nikkei['Open']
        indicepanel['Price'] = spy['Open']

        indicepanel = indicepanel.fillna(method='ffill')
        indicepanel = indicepanel.dropna()

        # Find the indice with largest correlation
        corr_array = indicepanel.iloc[:, :-1].corr()['target_stock']
        corr_array1 = pd.DataFrame(corr_array)

        return corr_array1



    @staticmethod
    def layout_vis(stock):
        perd = '1d'
        end_realdata =  datetime.now().strftime("%Y") + '-' + \
                        datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d")

        start_date = (datetime.now()-timedelta(days=5))
        start_traindateD = start_date.strftime("%Y") + '-' + \
                           start_date.strftime("%m") + '-'  + start_date.strftime("%d")

        aord = DataAccess.data_auto_reader('^AORD',start_traindateD,end_realdata,perd)
        nikkei = DataAccess.data_auto_reader('1360.T',start_traindateD,end_realdata,perd)
        hsi = DataAccess.data_auto_reader('^HSI',start_traindateD,end_realdata,perd)

        daxi = DataAccess.data_auto_reader('DAX',start_traindateD,end_realdata,perd)
        cac40 = DataAccess.data_auto_reader('^FCHI',start_traindateD,end_realdata,perd)

        aord['Diff_price'] = aord.iloc[-1]['Close'] - aord.iloc[-1]['Open']
        hsi['Diff_price'] = hsi.iloc[-1]['Close'] - hsi.iloc[-1]['Open']
        nikkei['Diff_price'] = nikkei.iloc[-1]['Close'] - nikkei.iloc[-1]['Open']


        data = [('aord',aord.iloc[-1]['Diff_price']),('hsi',hsi.iloc[-1]['Diff_price']),('nikkei',nikkei.iloc[-1]['Diff_price'])]

        hv_asia_bar = hv.Bars(data, hv.Dimension('Asian Market'), 'Price Diff',label=aord.iloc[-1]['Date'].strftime("%Y-%m-%d")).opts(height=600, width=700)


        return hv_asia_bar


    @staticmethod
    def layout_vis2(stock):
        perd = '1d'
        end_realdata =  datetime.now().strftime("%Y") + '-' + \
                        datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d")

        start_date = (datetime.now()-timedelta(days=5))
        start_traindateD = start_date.strftime("%Y") + '-' + \
                           start_date.strftime("%m") + '-'  + start_date.strftime("%d")


        daxi = DataAccess.data_auto_reader('DAX',start_traindateD,end_realdata,perd)
        cac40 = DataAccess.data_auto_reader('^FCHI',start_traindateD,end_realdata,perd)

        hv_plt_dax = hv.Curve((daxi.Date.tolist(), daxi.Close.tolist()), label='DAX').opts(
            height=600, width=700, xrotation= 90,line_color='r',xlabel='Date',ylabel='Price $',hooks=[StockIndex.apply_formatter])

        hv_plt_cac40 = hv.Curve((cac40.Date.tolist(), cac40.Close.tolist()), label='Cac40').opts(
            height=600, width=700, xrotation= 90,line_color='b',xlabel='Date',ylabel='Price $',hooks=[StockIndex.apply_formatter])



        layout = hv.Layout(hv_plt_dax * hv_plt_cac40)

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



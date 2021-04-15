import panel as pn
from lambda_stock_ui import Page
import numpy as np
from panel.interact import interact
import pandas as pd
from datetime import datetime,timedelta
import holidays
from data_access_layer import DataAccess
from scipy.stats import norm


class FinancialAnalysis(Page):

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
                # Financial Analysis
                """,background='#007500')

        title.session_id = self.session_id

        output = interact(self.statistics_model, stock='TSLA')
        output.session_id = self.session_id
        df_info = output[1][0].object
        # df_widget = pn.widgets.DataFrame(
        #     df_info, name='DataFrame', autosize_mode='fit_columns',width=900)
        # df_widget.session_id = self.session_id

        app_col = pn.Column(title,output[0], output[1][0],width=900)

        return app_col


    @staticmethod
    def statistics_model(stock):
        perd = '1d'
        end_traindate = '2021-01-01'
        start_traindateD = '2020-01-01'
        end_realdata =  datetime.now().strftime("%Y") + '-' + \
                        datetime.now().strftime("%m") + '-'  + datetime.now().strftime("%d")

        ticker_df = DataAccess.data_auto_reader(stock,start_traindateD,end_realdata,perd)


        # statistical analysis
        ticker_df['LogReturn'] = np.log(ticker_df['Close']).shift(-1) - np.log(ticker_df['Close'])

        ticker_df['MA10'] = ticker_df['Close'].rolling(10).mean() #fast signal
        ticker_df['MA50'] = ticker_df['Close'].rolling(50).mean() #slow signal

        ticker_df['MA40'] = ticker_df['Close'].rolling(40).mean() #fast signal
        ticker_df['MA200'] = ticker_df['Close'].rolling(200).mean() #slow signal

        ticker_df['Shares'] = [1 if ticker_df.loc[ei, 'MA10']>ticker_df.loc[ei, 'MA50'] else 0 for ei in ticker_df.index]

        ticker_df['Shares2'] = [1 if ticker_df.loc[ei, 'MA40']>ticker_df.loc[ei, 'MA200'] else 0 for ei in ticker_df.index]

        #daily profit
        ticker_df['Close1'] = ticker_df['Close'].shift(-1) #tomorrow's price
        ticker_df['Profit'] = [ticker_df.loc[ei,'Close1']-ticker_df.loc[ei,'Close'] if ticker_df.loc[ei,'Shares']==1
                           else 0 for ei in ticker_df.index]

        ticker_df['Profit2'] = [ticker_df.loc[ei,'Close1']-ticker_df.loc[ei,'Close'] if ticker_df.loc[ei,'Shares2']==1
                               else 0 for ei in ticker_df.index]

        ticker_df['wealth'] = ticker_df['Profit'].cumsum()
        ticker_df['wealth2'] = ticker_df['Profit2'].cumsum()

        Total_money_make = ticker_df.loc[ticker_df.index[-2],'wealth']
        Total_money_make2 = ticker_df.loc[ticker_df.index[-2],'wealth2']
        Total_money_invest = ticker_df.loc[ticker_df.index[0],'Close']

        #approximate mean and variance of the log daily return
        mu = ticker_df['LogReturn'].mean()
        sigma = ticker_df['LogReturn'].std(ddof=1)

        # to calculate this probability (losing over 5%), we use CDF
        prob_drop1 = norm.cdf(-.05,mu,sigma)

        # how about probability of dropping over 40% in 1 year (220 trading days)?
        # assumption: daily returns are independent --> wrong simplified assumption
        mu220 = 220*mu
        sigma220 = 220**.5*sigma
        prob_drop2 = norm.cdf(-.4,mu220,sigma220)

        # Quantiles, value at risk (VaR), measure how much a set of investment might lose
        daily_return_percent = norm.ppf(0.05,np.exp(mu),np.exp(sigma)) # VaR = 5% quantile, VaR at the level 95%
        # it means with 5% chance, the daily return is worse than daily_return_percent*100 %
        daily_return_percent2 = norm.ppf(0.4,np.exp(mu),np.exp(sigma))

        # 3-1: Confidence interval
        # values for calculating the 80% confidence interval
        z_left = norm.ppf(0.1)
        z_right = norm.ppf(0.9)
        sample_mean = ticker_df['LogReturn'].mean()
        sample_std = ticker_df['LogReturn'].std(ddof=1)/(ticker_df.shape[0])**0.5

        interval_left = sample_mean+z_left*sample_std
        interval_right = sample_mean+z_right*sample_std

        sample_mu_return = np.exp(sample_mean)
        lower_interval = np.exp(interval_left)
        upper_interval = np.exp(interval_right)

        # hypothesis testing
        xbar = ticker_df['LogReturn'].mean()
        s = ticker_df['LogReturn'].std(ddof=1)
        n = ticker_df['LogReturn'].shape[0]
        zhat = (xbar-0)/(s/(n**.5))

        # two tails TEST
        alpha=0.05
        zleft = norm.ppf(alpha/2,0,1)
        zright = -zleft
        Decision_2tail = zhat>zright or zhat<zleft


        # one tail TEST
        alpha=0.05
        zright = norm.ppf(1-alpha,0,1)
        Decision_1tail = zhat>zright

        #calculate p value for two tails test
        alpha=0.05
        p_value = 1 - (norm.cdf(abs(zhat),0,1))
        Decision_Pvalue = p_value<alpha


        stats_ouput = pd.DataFrame({
            "Item": ["Sample mean of daily return","80% confid Lower-value (daily return)","80% confid Upper-value (daily return)","P daily loss over 5%","P dropping over 40%/year","VAR: 5% chance Daily return worse than:","VAR: 40% chance Daily return worse than:","Total money made (based on MA10/MA50)","Total money made (based on MA40/MA200)","Total money invested","Start Date","End Date","reject zero return hypothesis at significant level of " + str(int(alpha*100)) + "%","reject zero return hypothesis at significant level of " + str(int(alpha*100)) + "%","reject zero return hypothesis at significant level of " + str(int(alpha*100)) + "%"],
            "Value": [sample_mu_return,lower_interval,upper_interval,prob_drop1,prob_drop2,daily_return_percent,daily_return_percent2,Total_money_make,Total_money_make2,Total_money_invest,start_traindateD,end_realdata,Decision_2tail,Decision_1tail,Decision_Pvalue]
        })

        return stats_ouput


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
        date_list = FinancialAnalysis.workdays(np.min(data_pred.Date),np.max(data_pred.Date))
        business_day = pd.DataFrame(np.array(date_list),columns=['Date'])
        data_pred1 = pd.merge(business_day,data_pred, on ='Date')
        data_pred1 = data_pred1[['Date','Price']]

        ind_drop = []
        for i in range(0,len(drop_dates2)):
            ind_drop.append(data_pred1[data_pred1.Date ==  drop_dates2[i]].index[0])

        data_pred1 = data_pred1.drop(ind_drop)
        return data_pred1
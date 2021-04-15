from data_access_layer import DataAccess

ticker_symbol = 'TSLA'
start_date = '2010-11-01'
end_date = '2021-01-01'
perd = '1mo'
ticker_df = DataAccess.data_auto_reader(ticker_symbol,start_date,end_date,perd)

tsla,date = DataAccess.data_cleaner(ticker_df)

tslam = DataAccess.data_publisher(ticker_df,perd)

train, test = DataAccess().ts_train_test_split(tsla)
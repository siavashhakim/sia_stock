from anomaly_detection import AnomalyDetection
import numpy as np
from data_access_layer import DataAccess

ticker_symbol = 'TSLA'
start_date = '2010-11-01'
end_date = '2021-01-01'
perd = '1mo'
ticker_df = DataAccess.data_auto_reader(ticker_symbol,start_date,end_date,perd)
tsla,date = DataAccess.data_cleaner(ticker_df)


ts_matrix = [tsla, tsla]
events_real = np.zeros(len(tsla)) # no real event assumption

events = AnomalyDetection(tsla,events_real,ts_matrix).unsupervised_event_detection()

events2 = AnomalyDetection(tsla,events_real,ts_matrix).event_detection()

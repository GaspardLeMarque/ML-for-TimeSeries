#Use pandas_datareader lib to get CMC200 index
from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
import pandas as pd 
from datetime import datetime 

#Check the start date manually
start_date = '2018-12-31'
end_date = str(datetime.now().strftime('%Y-%m-%d'))

my_ticker = '%5ECMC200' 

def GetData(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
        print(stock_data)

    except RemoteDataError:
        print('No data for {t}'.format(t=ticker))

GetData(my_ticker)

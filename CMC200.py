#Script to get CMC200 index
from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
import pandas as pd 
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

start_date = '2018-12-31'
end_date = str(datetime.now().strftime('%Y-%m-%d'))

my_ticker = '%5ECMC200' 

def GetData(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
        print(stock_data)
        stock_data.to_csv('CMC200.csv') 

    except RemoteDataError:
        print('No data for {t}'.format(t=ticker))

GetData(my_ticker)

#Data and EDA
df = pd.read_csv("CMC200.csv", index_col='Date') 

#Plot the Adj Close price
df['Adj Close'].plot(label='CMC200', legend=True)
plt.show()  

#Histogram of the daily price change percent of Adj Close price
df['Adj Close'].pct_change().plot.hist(bins=100)
plt.xlabel('Adjusted close 1-day percent change')
plt.show()

#Kernel Density Estimation
sns.kdeplot(df['Adj Close'], shade=True)

#Correlations
#Calculate returns for the current day and for 5 days in the future
df['5d_future_close'] = df['Adj Close'].shift(-5)
df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
df['5d_close_pct'] = df['Adj Close'].pct_change(5)

#Calculate the correlation matrix between returns
corr = df[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr) #No autocorrelation

#Scatter plot of the current 5-day percent change vs the future 5-day percent change
plt.scatter(df['5d_close_pct'], df['5d_close_future_pct'])
plt.show() #No autocorrelation

#Create a pairplot 
sns.pairplot(df, vars=['5d_close_pct', '5d_close_future_pct'], diag_kind = 'kde', 
             plot_kws = {'alpha': 0.8, 's': 80, 'edgecolor': 'k'})

#Regression plot
sns.regplot('5d_close_pct', '5d_close_future_pct', df) #No autocorrelation

#Split into repeatable train and test sets
df_train, df_test = train_test_split(df['Adj Close'], test_size=0.33, random_state=123)
print(df_train.shape, df_test.shape)

from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
import pandas as pd 
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import torch
import os
import numpy as np
from tqdm import tqdm #Show loops progress bars
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

#Data
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

df = pd.read_csv("CMC200.csv", index_col='Date') 

#Setting the seed
np.random.seed(123)
torch.manual_seed(123)

#EDA
diff_p = df['Adj Close'].diff()
sns.lineplot(data=diff_p)

#Preprocessing
df_train, df_test = train_test_split(df['Adj Close'], test_size=0.33, random_state=123)
print(df_train.shape, df_test.shape) #318, 158

#Scale data to increase the training speed
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(df_train, axis=1))
train_data = scaler.transform(np.expand_dims(df_train, axis=1))
test_data = scaler.transform(np.expand_dims(df_test, axis=1))

#Divide TS to the sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

#Each sequence contains 5 data entries 
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

#Check the constructed tensors
X_train.shape
X_train[:2]
y_train.shape
y_train[:2]

#The model
class PricePredictor(nn.Module):
#Constructor, create the layers  
  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(PricePredictor, self).__init__()
    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers
    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )
    self.linear = nn.Linear(in_features=n_hidden, out_features=1)
#Reset the state after each example (stateless LSTM)
  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )
#Pass sequences into the LSTM layers 
  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
    

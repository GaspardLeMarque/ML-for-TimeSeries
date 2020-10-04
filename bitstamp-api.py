import requests
import json
import pandas as pd
import datetime
import numpy as np

# extracting BTCUSD
# API-endpoint 
URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
 
# define params for the API 
PARAMS = {'step':86400, 'limit':1000} #daily
  
# save the GET request as a response object 
response = requests.get(url = URL, params = PARAMS) 
  
# extract data in json format 
data = response.json() 
# (optionally) show data as a text
data = response.text

# convert the string to JSON  
parsed = json.loads(data)

# extract data
parsed["data"]["ohlc"][0]["close"] #observe a single value

for i in parsed["data"]["ohlc"]: #observe all columns
    print(i['timestamp'],i['high'],i['low'],i['open'],i['close'],i['volume'])

# fill in a DF with the extracted data
df = pd.DataFrame(parsed["data"]["ohlc"])   

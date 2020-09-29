import numpy as np
import pandas as pd
import glob
from functools import reduce

path = r'D:\Python\data\Currency\Monthly' 
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col='DATE')
    li.append(df)

df_merged = reduce(lambda left,right: pd.merge(left,right,on=['DATE'],
                                            how='outer'), li)

df_merged.rename(
   columns={'EXUSAL': 'AUD/USD',    #Australian Dollar
            'EXBZUS': 'USD/BRL',    #Brazilian Real   
            'EXCHUS': 'USD/CNY',    #Chinese Yuan
            'EXDNUS': 'USD/DKK',    #Danish Krone   
            'EXUSEU': 'EUR/USD',    #Euro    
            'EXUSUK': 'GBP/USD',    #Great British Pound
            'EXHKUS': 'USD/HKD',    #Hong Kong Dollar
            'EXINUS': 'USD/INR',    #Indian Rupee
            'EXKOUS': 'USD/KRW',    #South Korean Won
            'EXSLUS': 'USD/LKR',    #Sri Lankan Rupee
            'EXMXUS': 'USD/MXN',    #Mexican Peso
            'EXMAUS': 'USD/MYR',    #Malaysian Ringgit
            'EXNOUS': 'USD/NOK',    #Norwegian Krone
            'EXUSNZ': 'NZD/USD',    #New Zealand Dollar
            'EXSDUS': 'USD/SEK',    #Swedish Krona
            'EXTHUS': 'USD/THB',    #Thai Baht
            'EXTAUS': 'USD/TWD',    #Taiwan New Dollar
            'EXCAUS': 'USD/CAD',    #Canadian Dollar
            'EXSZUS': 'USD/CHF',    #Swiss Franc
            'EXJPUS': 'USD/JPY',    #Japanese Yen
            'EXSIUS': 'USD/SGD',    #Singapore Dollar
            'EXVZUS': 'USD/VEF',    #Venezuelan Bolivar
            'EXSFUS': 'USD/ZAR'     #South African Rand
            }, inplace=True)

pd.DataFrame.to_csv(df_merged, 'monthly_all.csv', sep=',', index='DATE')

import matplotlib.pyplot as plt

df = pd.read_csv('monthly_all.csv', header=0, index_col=0, parse_dates=True)
#header=0 to be able to replace col names
print(df.head())
df.dtypes #Check the data types of columns
data = {'Mean':df.mean(), 
        'Median':df.median(), 
        'Min':df.min(), 
        'Max':df.max()}
df1 = pd.DataFrame(data) 
df2 = pd.DataFrame(df.quantile([0.25,0.75])).transpose()

sixNumSmry = df1.join(df2)

#AUD/USD pair
df1 = df[['AUD/USD']] 
df1.plot(label='AUD', legend=True)
plt.xlabel("Months")
plt.ylabel("Price")
plt.show()  
plt.clf() 

#Build a histogram to inspect the distribution
df1.pct_change().plot.hist(bins=50)
plt.xlabel('Monthly percent change')
plt.show() #Skewed distribution

#Correlations between pairs
corr = df.corr()
print(corr)

#Plot heatmap of a correlation matrix
plt.subplots(figsize=(8,8))
sns.heatmap(corr, annot= True, linewidths=.5, annot_kws = {"size": 7})
plt.yticks(rotation=0, size = 10); plt.xticks(rotation=90, size = 10)    
plt.tight_layout() 
plt.show()

#Choose the pairs with the high correlation
corrList = corr.unstack().sort_values(kind="quicksort")
print(corrList[-29:-23]) #Pairs with corr more than 0.95

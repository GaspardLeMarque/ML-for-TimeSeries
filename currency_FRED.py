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

pd.DataFrame.to_csv(df_merged, 'monthly_all.csv', sep=',', na_rep='.', index='DATE')







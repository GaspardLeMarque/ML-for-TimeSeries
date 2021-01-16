#Script to get CMC200 index
from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
import pandas as pd 
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import talib #Add features
import statsmodels.api as sm
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid #Tune hyperparameters
from sklearn.ensemble import GradientBoostingRegressor

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

#Correlations
#Calculate returns for the current day and for 5 days in the future
df['5d_future_close'] = df['Adj Close'].shift(-5)
df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
df['5d_close_pct'] = df['Adj Close'].pct_change(5)

#Kernel Density Estimation
sns.kdeplot(df['Adj Close'], shade=True)
plt.clf()

#Calculate the correlation matrix between returns
corr = df[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr) #No autocorrelation

#Scatter plot of the current 5-day percent change vs the future 5-day percent change
plt.scatter(df['5d_close_pct'], df['5d_close_future_pct'])
plt.show() #No autocorrelation
plt.clf()

#Create a pairplot 
sns.pairplot(df, vars=['5d_close_pct', '5d_close_future_pct'], diag_kind = 'kde', 
             plot_kws = {'alpha': 0.8, 's': 80, 'edgecolor': 'k'})

#Regression plot
sns.regplot('5d_close_pct', '5d_close_future_pct', df) #No autocorrelation

#Multivariate density estimation
sns.boxplot(df['Adj Close'])
plt.clf()
sns.boxplot(df['5d_close_pct']) #has less variance, less right-skewed



#Returns 
df['5d_close_pct'].plot()
plt.clf()
df['5d_close_future_pct'].plot()

#Create targets and features
feature_names = ['5d_close_pct'] 

for n in [14, 30, 50, 200]:
    
    #Create the moving average indicator and normalize it by Adj Close
    df['ma' + str(n)] = talib.SMA(df['Adj Close'].values,
                              timeperiod=n) / df['Adj Close']
    #Create the RSI indicator
    df['rsi' + str(n)] = talib.RSI(df['Adj Close'].values, timeperiod=n)
    
    #Add RSI and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

print(feature_names)

#Drop all NA values
df.isnull().sum() #New cols have nulls
df = df.dropna()

features = df[feature_names]
targets = df['5d_close_future_pct']

#Create DataFrame from target column and feature columns
feature_and_target_cols = ['5d_close_future_pct'] + feature_names
feat_targ_df = df[feature_and_target_cols]

#Plot a correlation matrix of features and targets
corr = feat_targ_df.corr()
print(corr)

plt.subplots(figsize=(8,5))
sns.heatmap(corr, annot= True, annot_kws = {"size": 8})
plt.yticks(rotation=0, size = 10); plt.xticks(rotation=90, size = 10) #Fix ticklabel directions and size  
plt.tight_layout()  
plt.show()
plt.clf()

#Create a scatter plot of the most highly correlated variable with the target
#Check the heatmap first and then create a plot
sns.regplot(df['ma50'], df['5d_close_future_pct']) #Highest corr = 0.18 (on 19.12)

#Linear model
#Add a constant to the features
linear_features = sm.add_constant(features)

#Split the whole DF into repeatable train and test sets (as an option)
df_train, df_test = train_test_split(df['Adj Close'], test_size=0.33, random_state=123)
print(df_train.shape, df_test.shape)

#Create a size for the training set that is 85% by hand
train_size = int(0.85 * features.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]

#Fit a linear model
model = sm.OLS(train_targets, train_features)
results = model.fit()  
print(results.summary())

#Features with p <= 0.05 are typically considered significantly different from 0
print(results.pvalues) #All ftrs except rsi14 have p val <= 0.05

#Make predictions from the model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

#Results evaluation
#Scatter the predictions vs the targets 
plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha=0.2, color='r', label='test')

xmin, xmax = plt.xlim() #Add the prediction line
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  
plt.show()
plt.clf()

#Decision trees
#Create a decision tree regression model with default arguments
decision_tree = DecisionTreeRegressor()

#Fit the model to the training features and targets
decision_tree.fit(train_features, train_targets)

#Check the score on train and test sets
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))

#Loop through a few different max depths and check the performance
for d in [3, 5, 10]:
    #Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth = d) #max_depth - the N of splits in the tree
    decision_tree.fit(train_features, train_targets)

    #Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(train_features, train_targets))
    print(decision_tree.score(test_features, test_targets), '\n')
    #Depth=10 shows the smallest value in the test set, 
    #but the R^2 is too small, so the model is wrong

#Check the results
#Use the best max_depth of 10 to fit a decision tree
decision_tree = DecisionTreeRegressor(max_depth=10)
decision_tree.fit(train_features, train_targets)

#Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

#Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
plt.show() #There is no linear dependency in the test set
plt.clf()

#Random forests (balance btwn high variance and high bias models)
#Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

#Look at the R^2 scores on train and test
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets)) #Unacceptably low R^2

#Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4, 8], 'random_state': [123]}
test_scores = []

#Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

#Find best hyperparameters from the test score
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])

#Evaluate performance
#Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=123)
rfr.fit(train_features, train_targets)

#Make predictions with the model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

#Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show() #There is also no linear dependency in the test set
plt.clf()

#Gradient boosting model
#Create GB model 
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=123)
gbr.fit(train_features, train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets)) #Again the score is too low
#Perhaps parameters are wrong

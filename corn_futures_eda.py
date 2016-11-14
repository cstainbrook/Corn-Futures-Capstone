import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.dates import YearLocator
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from math import log
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.transforms as mtransforms


def plot_hist_feature(feature):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.hist(df[feature][0:10308], bins=20, color='k', alpha=.6)
    plt.axvline(np.mean(df[feature]), color='r', alpha=.7)
    plt.suptitle(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_corn_prices():
    y = df['Settle']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    corn_syrup = datetime(1970,1,1)
    e85 = datetime(2000,1,1)
    y_avg = movingaverage(y, 504)
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, color='gray', alpha=.6, label='Price')
    ax.plot(x[252:-252], y_avg[252:-252], color='r', alpha=.7, label='2-Year Moving Average')
    ax.plot((corn_syrup, corn_syrup), (min(y),200), label='Corn Syrup\'s Commercial Introduction', color='k')
    ax.plot((e85, e85), (min(y),300), label='E85 Becomes Widely Used', color='k')
    plt.suptitle('Corn Prices Over Time:\n1959 - 2016', size=20, alpha=.8)
    plt.ylabel('Price of Nearest Futures Contract($)')
    plt.legend(loc=2)
    plt.show()

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_daily_change():
    df['daily change'] = df['Settle'].diff() * -1
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    mu = df['daily change'].mean()
    std = df['daily change'].std()

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    # ax.hist(df['daily change'][1:], bins=50, range=[-70, 50], color='red')
    ax.plot(x, df['daily change'])
    # ax.hist(np.random.normal(mu, std, len(df)-1), bins=50, range=[-70, 50], color='k', alpha=.3)
    plt.suptitle('Daily Futures Price Change', size=20, alpha=.8)
    plt.title('(Outliers Excluded)', position=(.5, 1.01))
    plt.xlabel('Price Change ($)')
    plt.ylabel('Frequency')
    plt.show()

def plot_corn_log():
    df['log_corn'] = df['Settle'].map(lambda x: log(x))

    y = df['log_corn']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, color='gray', alpha=.6, label='Price')
    plt.suptitle('Log Corn Prices Over Time:\n1959 - 2016', size=20, alpha=.8)
    plt.ylabel('Log Price of Nearest Futures Contract($)')
    plt.legend(loc=2)
    # plt.show()

def plot_corn_prices_adjusted():
    y = df['Inflation Adjusted Price']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    corn_syrup = datetime(1970,1,1)
    e85 = datetime(2000,1,1)
    y_avg = movingaverage(y, 504)

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    loc = YearLocator(5)
    ax.plot(x, y, color='gray', alpha=.6, label='Inflation Adjusted Price')
    ax.plot(x[252:-252], y_avg[252:-252], color='r', alpha=.7, label='2-Year Moving Average')
    ax.plot((corn_syrup, corn_syrup), (0,1500), color='k')
    ax.plot((e85, e85), (0,1500), color='k')
    ax.xaxis.set_major_locator(loc)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.text(corn_syrup + timedelta(days=100), 50, 'Corn Syrup\'s\nCommerical Introduction')
    plt.text(e85 + timedelta(days=100), 50, 'E85 Becomes\nWidely Used')
    plt.suptitle('Corn Prices Over Time:\n1959 - 2016', size=20, alpha=.8)
    plt.ylabel('Inflation Adjusted\nPrice of Nearest Futures Contract ($)', size=14, labelpad=30)
    plt.legend()
    # plt.show()
    plt.savefig('Figures/inflation_adjusted_demand_shocks.png')

def plot_nino_adjusted():
    y = df['Inflation Adjusted Price']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    x = np.array([x.to_pydatetime() for x in x])
    y_avg = movingaverage(y, 504)
    nino_mask = (df['ONI Index'] > 0).values

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    loc = YearLocator(5)
    plt.tick_params(axis='both', which='major', labelsize=11)
    ax.plot(x, y, color='gray', alpha=.6, label='Inflation Adjusted Price')
    ax.plot(x[252:-252], y_avg[252:-252], color='r', alpha=.7, label='2-Year Moving Average')
    ax.fill_between(x, np.zeros(len(x)), y, where=nino_mask, alpha=.6, label='El Nino Years')
    ax.xaxis.set_major_locator(loc)
    plt.suptitle('Corn Prices Over Time:\n1959 - 2016', size=20, alpha=.8)
    plt.ylabel('Inflation Adjusted\nPrice of Nearest Futures Contract ($)', size=14, labelpad=30)
    plt.legend()
    # plt.show()
    plt.savefig('Figures/inflation_adjusted_nino.png')

def plot_adjusted_log():
    df['adjusted_log_corn'] = df['Inflation Adjusted Price'].map(lambda x: log(x))

    y = df['adjusted_log_corn']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, color='gray', alpha=.6, label='Price')
    plt.suptitle('Log Corn Prices (Inflation Adjusted) Over Time:\n1959 - 2016', size=20, alpha=.8)
    plt.ylabel('Inflation Adjusted Price of Nearest Futures Contract (Log $)')
    plt.legend(loc=2)
    plt.show()

def plot_log_daily_change():
    df['daily change'] = df['Inflation Adjusted Price'].diff() * -1
    y = np.append(df['daily change'][0:10901].values, df['daily change'][10903:].values)
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    x = np.append(x.values[1:10901], x.values[10903:])
    y_ma = pd.rolling_mean(y, window=252)
    y_rolling_std = pd.rolling_std(y, window=252)

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y[1:])
    ax.plot(x, y_ma[1:])
    ax.plot(x, y_rolling_std[1:])
    plt.suptitle('Daily Futures Log Price Change', size=20, alpha=.8)
    plt.title('(July 1973 Outlier Excluded)', position=(.5, 1.01))
    plt.xlabel('Date')
    plt.ylabel('Price Change (Log $)')
    plt.show()

def perform_dickey_fuller():
    df['daily change'] = df['Inflation Adjusted Price'].diff() * -1
    y = np.append(df['daily change'][0:10901].values, df['daily change'][10903:].values)

    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(y[1:], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def plot_production_price():
    y = df['Inflation Adjusted Price']
    x = df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    x = np.array([x.to_pydatetime() for x in x])
    y_avg = movingaverage(y, 504)
    production_df = pd.read_csv('USDA Data/usda_production.csv')[['Year', 'Period', 'Value']]
    production_df = production_df[production_df['Period'] == 'YEAR']
    production_df['Date'] = production_df['Year'].map(lambda x: datetime(x, 1, 1))
    production_df.set_index('Date', inplace=True)
    x_prod = production_df.index[0:58].values
    prod = production_df['Value'].values[0:58]
    prod = [int(''.join(val.split(','))) for val in prod]
    prod = [val/1000000 for val in prod]

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    loc = YearLocator(5)
    plt.tick_params(axis='both', which='major', labelsize=11)
    # ax.plot(x, y, color='gray', alpha=.6, label='Inflation Adjusted Price')
    ax.plot(x[252:-252], y_avg[252:-252], color='r', alpha=.7, label='2-Year Moving Average Price')
    plt.legend(loc=2)
    plt.ylabel('Inflation Adjusted Price ($)', size=14, labelpad=30)
    ax2 = fig.add_subplot(111, sharex=ax, frameon=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.plot(x_prod, prod, color='k', label='Production')
    plt.ylabel('Corn Production (Million Bushels)', size=14, labelpad=30)
    ax.xaxis.set_major_locator(loc)
    plt.suptitle('Corn Prices vs. Production:\n1959 - 2016', size=20, alpha=.8)
    plt.legend(loc=0)
    plt.savefig('Figures/price_vs_production.png')


if __name__ == '__main__':
    df = pd.read_csv('full_database.csv')
    # plot_hist_feature('Supply Level')
    # plot_corn_prices()
    # plot_daily_change()
    # plot_corn_log()
    # plot_corn_prices_adjusted()
    # plot_nino_adjusted()
    # plot_adjusted_log()
    # plot_log_daily_change()
    # perform_dickey_fuller()
    plot_production_price()

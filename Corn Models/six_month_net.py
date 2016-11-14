import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta as td
import seaborn as sns
import pickle
from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.ext.CommissionReport import CommissionReport
from ib.ext.TickType import TickType as tt
from time import sleep, strftime

np.random.seed(42)

class PriceInformation(object):
    def __init__(self, symbol, sec_type, expiry, exch, curr):
        self.symbol = symbol
        self.sec_type = sec_type
        self.expiry = expiry
        self.exch = exch
        self.curr = curr
        self.contract = self.make_contract()
        self.conn = None

    def make_contract(self):
        Contract.m_symbol = self.symbol
        Contract.m_secType = self.sec_type
        Contract.m_expiry = self.expiry
        Contract.m_exchange = self.exch
        Contract.m_currency = self.curr
        return Contract

def current_price_handler(msg):
    current_price = msg.price

def hist_data_handler(msg):
    prices.append(msg.close)

def all_handler(msg):
    print(msg)

def check_business_day(datetime_object):
    if datetime_object + BDay(0) == datetime_object:
        return True
    else:
        return False

def get_current_prices():
    conn = ibConnection(port=7497, clientId=110)
    conn.registerAll(all_handler)
    conn.unregister(all_handler, message.historicalData)
    conn.register(hist_data_handler, message.historicalData)
    conn.unregister(all_handler, 'TickPrice')
    conn.register(current_price_handler, 'TickPrice')
    conn.connect()

    month = datetime.today().month
    year = datetime.today().year
    # shortest_future_date =
    corn_future = PriceInformation('ZC', 'FUT', '20161214', 'ECBOT', 'USD')
    endatetimeime = strftime('%Y%m%d %H:%M:%S')
    conn.reqHistoricalData(1,corn_future.contract,endatetimeime,'4 D','1 day','TRADES',1,1)
    sleep(5)
    conn.disconnect()

def train_test_split(df, trainsize=.67):
    train_size = int(len(df)*trainsize)
    train_df = df[0:train_size]
    test_df = df[train_size:]
    return train_df, test_df

def create_neural_net(X, y):
    model = Sequential()
    model.add(Dense(6, input_dim=10))
    model.add(Dense(2, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, nb_epoch=100)
    return model

def make_preds(X, y, model):
    preds = model.predict(X)
    preds = scaler_y.inverse_transform(preds)
    y = scaler_y.inverse_transform(y)
    model_score = math.sqrt(mean_squared_error(y, preds[:,0]))
    return preds, model_score

def plot_predictions():
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    dates_plot = np.array([x.to_pydatetime() for x in dates])
    train_preds_plot = train_preds[0].ravel()
    test_preds_plot = np.append(np.full(len(train_preds[0]), np.nan), test_preds[0]).ravel()

    ax.plot(dates, scaler_y.inverse_transform(six_df[['Settle']]), color='k', label='True Price')


    ax.plot(dates[:len(train_preds[0])], train_preds[0], color='grey', label='In-Sample Predictions')
    ax.fill_between(dates_plot[:len(train_preds[0])], train_preds_plot, (train_preds_plot + 2*train_preds[1]), color='grey', alpha=.6)
    ax.fill_between(dates_plot[:len(train_preds[0])], train_preds_plot, (train_preds_plot + -2*train_preds[1]), color='grey', alpha=.6)


    ax.plot(dates, test_preds_plot, color='r', label='Out-of-Sample Predictions')
    ax.fill_between(dates_plot, test_preds_plot, (test_preds_plot + 2*test_preds[1]), color='red', alpha=.4)
    ax.fill_between(dates_plot, test_preds_plot, (test_preds_plot + -2*test_preds[1]), color='red', alpha=.4)

    plt.legend(loc=2)
    plt.ylabel('Corn Futures Price ($)', size=14, labelpad=30)
    plt.suptitle('Neural Network Predictions\nUsing 6 Month Lagged Features', size=20, alpha=.8)
    # plt.show()
    plt.savefig('../Figures/neuralnet_six_month.png')

def make_current_pred():
    current_price = prices[-3]
    day_of_year = ((datetime.today() + td(days=127)) - datetime((datetime.today()+td(days=127)).year - 1, 12, 31)).days

    #Format is Day of Year, Corn Syrup, E85, Price 6 Months Ago, Supply Six Months Ago, and ONI Six Months Ago
    current_inputs = np.array([[day_of_year, current_price, current_df.ix[0, 'Supply Level'], current_df.ix[0, 'ONI Index'], current_df.ix[0, 'Iowa 6 Month Precip'], current_df.ix[0, 'Iowa 6 Month Temp'],\
     current_df.ix[0, 'Illinois 6 Month Precip'], current_df.ix[0, 'Illinois 6 Month Temp'], current_df.ix[0, 'Nebraska 6 Month Precip'], current_df.ix[0, 'Nebraska 6 Month Temp']]])
    current_inputs = scaler_x.transform(current_inputs)

    six_month_pred = model.predict(current_inputs)
    six_month_pred = scaler_y.inverse_transform(six_month_pred)
    return six_month_pred

def plot_current_pred():
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(.5, 1.51, .01)
    upper_line = (prices[-3]-((current_upper_bound-prices[-3])/1.01*1.5)) + ((current_upper_bound-prices[-3])/1.01)*(1+x)
    lower_line = (prices[-3]-((current_lower_bound-prices[-3])/1.01*1.5)) + ((current_lower_bound-prices[-3])/1.01)*(1+x)
    pred_line = (prices[-3]-((current_prediction-prices[-3])/1.01*1.5)) + ((current_prediction-prices[-3])/1.01)*(1+x)


    ax.plot(x, upper_line[0], color='k', alpha=.2)
    ax.plot(x, lower_line[0], color='r', alpha=.2)
    ax.plot(x, pred_line[0], color='k')
    ax.fill_between(x, pred_line[0], upper_line[0], color='grey', alpha=.6)
    ax.fill_between(x, lower_line[0], pred_line[0], color='r', alpha=.4)
    ax.scatter(1.5, current_prediction, color='k', s=70)
    ax.annotate('${}'.format(round(current_prediction[0][0],2)), xy=(1.5, current_prediction), xytext=(1.52, current_prediction-10))
    ax.annotate('${}'.format(round(current_upper_bound[0][0],2)), xy=(1.5, current_upper_bound), xytext=(1.52, current_upper_bound-10))
    ax.annotate('${}'.format(round(current_lower_bound[0][0],2)), xy=(1.5, current_lower_bound), xytext=(1.52, current_lower_bound-10))
    ax.annotate('${}'.format(round(prices[-3],2)), xy=(0.5, prices[-3]), xytext=(0.38, prices[-3]-5))
    plt.ylim(0, 800)
    plt.xlim(.35, 1.65)
    plt.suptitle('Current Prediction for 6 Months Out', size=20, alpha=.8)
    plt.ylabel('Price of Nearest Corn Futures Contract ($)', size=14, labelpad=30)
    plt.xticks([.5, 1.5], ['Current Price', 'Predicted Price\n6 Months Out'], rotation=20, size=12)
    plt.yticks([],[])
    # plt.show()
    plt.savefig('../Figures/current_pred_six.png')

def plot_gains_losses(gains_losses):
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.hist(gains_losses, bins=100, color='r', alpha=.7)
    plt.axvline(x=0, color='k')
    plt.ylabel('Frequency', size=14, labelpad=30)
    plt.xlabel('Profit/Loss', size=14, labelpad=10)
    plt.suptitle('Profit/Loss on All Trades', size=20, alpha=.8)
    plt.savefig('../Figures/profits_hist_six.png')
    # plt.show()

def trading_results(strategy='simple', discount=0, leverage=1):
    trading_df = current_df[['Date','Settle']]
    trading_df['Date'] = trading_df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    trading_df.set_index('Date', inplace=True)
    trading_df = trading_df[0:len(test_preds[0])]
    trading_df['Prediction'] = test_preds[0]
    trading_df['Price at Time of Prediction'] = current_df['Settle'][127:(127+len(test_preds[0]))].values
    if strategy=='simple':
        buy_mask = trading_df['Price at Time of Prediction'] < trading_df['Prediction']
        buy_profits = (trading_df[buy_mask]['Settle'] - trading_df[buy_mask]['Price at Time of Prediction'].values) * leverage
        sell_profits = (trading_df[~buy_mask]['Price at Time of Prediction'] - trading_df[~buy_mask]['Settle'].values) * leverage
        total_profit = sum(buy_profits) + sum(sell_profits)
        total_return = total_profit/sum(trading_df['Price at Time of Prediction'])
        annualized_return = (1+total_return)**(1/(len(trading_df)/365.)) - 1
        all_profits = np.append(buy_profits, sell_profits)
        max_loss = min(all_profits)
        plot_gains_losses(all_profits)
        return total_profit, total_return, annualized_return, max_loss

    if strategy=='discount':
        buy_mask = trading_df['Price at Time of Prediction'] < trading_df['Prediction']*(1-discount)
        sell_mask = trading_df['Price at Time of Prediction']*(1-discount) > trading_df['Prediction']
        buy_profits = (trading_df[buy_mask]['Settle'] - trading_df[buy_mask]['Price at Time of Prediction'].values) * leverage
        sell_profits = (trading_df[sell_mask]['Price at Time of Prediction'] - trading_df[sell_mask]['Settle'].values) * leverage
        total_profit = sum(buy_profits) + sum(sell_profits)
        total_return = total_profit/(sum(trading_df[buy_mask]['Price at Time of Prediction']) + sum(trading_df[sell_mask]['Price at Time of Prediction']))
        annualized_return = (1+total_return)**(1/((len(trading_df[buy_mask])+len(trading_df[sell_mask]))/365.)) - 1
        all_profits = np.append(buy_profits, sell_profits)
        max_loss = min(all_profits)
        plot_gains_losses(all_profits)
        return total_profit, total_return, annualized_return, max_loss

    if strategy=='hit_rate':
        buy_mask = trading_df['Price at Time of Prediction'] < trading_df['Prediction']*(1-discount)
        sell_mask = trading_df['Price at Time of Prediction']*(1-discount) > trading_df['Prediction']
        buy_hit_rate = sum((trading_df[buy_mask]['Settle'] - trading_df[buy_mask]['Price at Time of Prediction'].values)>0)
        sell_hit_rate = sum((trading_df[sell_mask]['Price at Time of Prediction'] - trading_df[sell_mask]['Settle'].values)>0)
        total_hit_rate = float((buy_hit_rate + sell_hit_rate))/(len(trading_df[buy_mask]) + len(trading_df[sell_mask]))
        total_trades = len(trading_df[buy_mask]) + len(trading_df[sell_mask])
        return total_hit_rate, total_trades


if __name__ == '__main__':
    six_df = pd.read_csv('../six_month_database.csv', index_col=0)
    #RMSE of Initial Model without Dollar, Soybeans, or Oil: 43.0(train), 136.4(test)
    #RMSE of Secondary Model with Dollar, Soybeans, and Oil: 39.2(train), 211.4(test)
    #RMSE of Tertiary Model with Soybeans: 42.8(train), 138.0(test)
    #RMSE of Model with USD Index: 40.3(train), 166.8(test)
    #RMSE of Model with Oil: 42.7(train), 159.7(test)
    #RMSE of Model without ONI: 42.9(train), 140.7(test)
    #RMSE of Linear Model without Corn Syrup, E85: 43.1(train), 128.8(test)
    #RMSE of Tanh Model without Corn Syrup, E85: 35.8(train), 198.9(test)
    #RMSE of Relu Model without Corn Syrup, E85: 36.0(train), 221.0(test)
    #RMSE of Sigmoid Model without Corn Syrup, E85: 35.4(train), 194.7(test)
    #RMSE of Linear Model with 3 layers: 43.1(train), 127.3(test)
    #RMSE of 3 Layer Model w/linear hidden, sigmoid output: 44.4(train), 122.1(test)
    #RMSE of 3 Layer Model w/linear hidden, sigmoid output, 8 node input, 2 node layer: 44.4(train), 118.9(test)
    #RMSE of previous with weather data included: 40.7(train), 110.8(test)
    #RMSE of previous with weather data included, 6 nodes: 40.4(train), 106.5(test)
    #RMSE of previous with weather data included and Minnesota, 6 nodes: (train), (test)


    six_df.drop(['USD Index 6 Months Ago', 'Oil Prices 6 Months Ago', 'Soybean Prices 6 Months Ago', 'Corn Syrup', 'E85', 'Minnesota Precip 6 Months Ago', 'Minnesota Temp 6 Months Ago'], axis=1, inplace=True)
    six_df = six_df[::-1]
    six_df.dropna(axis=0, inplace=True)
    dates = six_df.pop('Date').map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    six_df[['Day of Year', 'Price 6 Months Ago', 'Supply 6 Months Ago', 'ONI Index 6 Months Ago', 'Iowa Precip 6 Months Ago', 'Iowa Temp 6 Months Ago', 'Illinois Precip 6 Months Ago', 'Illinois Temp 6 Months Ago', 'Nebraska Precip 6 Months Ago', 'Nebraska Temp 6 Months Ago']] =\
     scaler_x.fit_transform(six_df[['Day of Year', 'Price 6 Months Ago', 'Supply 6 Months Ago', 'ONI Index 6 Months Ago', 'Iowa Precip 6 Months Ago', 'Iowa Temp 6 Months Ago', 'Illinois Precip 6 Months Ago', 'Illinois Temp 6 Months Ago', 'Nebraska Precip 6 Months Ago', 'Nebraska Temp 6 Months Ago']])
    six_df[['Settle']] = scaler_y.fit_transform(six_df[['Settle']])


    train_df, test_df = train_test_split(six_df)
    train_y = train_df.pop('Settle').values
    train_X = train_df.values
    test_y = test_df.pop('Settle').values
    test_X = test_df.values


    model = create_neural_net(train_X, train_y)
    train_preds = make_preds(train_X, train_y, model)
    test_preds = make_preds(test_X, test_y, model)
    plot_predictions()

    current_df = pd.read_csv('../full_database.csv')
    # prices=[]
    # get_current_prices()
    # current_prediction = make_current_pred()
    # current_lower_bound = current_prediction - test_preds[1]
    # current_upper_bound = current_prediction + test_preds[1]
    # plot_current_pred()

    total_profit, total_return, annualized_return, max_loss = trading_results(strategy='discount', discount=.1, leverage=15)
    total_hit_rate, total_trades = trading_results(strategy='hit_rate', discount=.3)

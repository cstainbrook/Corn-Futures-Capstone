import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime

np.random.seed(42)

def train_test_split(numpy_arr):
    train_size = int(len(numpy_arr) * 0.67)
    test_size = len(numpy_arr) - train_size
    train, test = numpy_arr[0:train_size], numpy_arr[train_size:len(numpy_arr)]
    return train, test

def create_lagged(numpy_arr, look_back=1):
    X, y = [], []
    for i in range(len(numpy_arr)-look_back-1):
        a = numpy_arr[i: (i+look_back)]
        X.append(a)
        y.append(numpy_arr[i + look_back])
    X = np.array(X)
    y = np.array(y).reshape(len(y), 1)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    # y = np.reshape(y, (y.shape[0], 1, y.shape[1]))
    return X, y

def make_net(X, y, num_neurons, input_dimension):
    model = Sequential()
    model.add(LSTM(num_neurons, input_dim=input_dimension))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, nb_epoch=100, batch_size=1, verbose=2)
    return model

def make_preds(X, y, model):
    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)
    y = scaler.inverse_transform(y)
    model_score = math.sqrt(mean_squared_error(y[:,0], preds[:,0]))
    return preds, model_score

def plot_predictions():
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(scaler.inverse_transform(prices_arr), color='k', label='True Price')
    ax.plot(train_preds[0], color='grey', label='In-Sample Predictions')
    test_preds_plot = np.append(np.full(len(train_preds[0]), np.nan), test_preds[0])
    ax.plot(test_preds_plot, color='r', label='Out-of-Sample Predictions')
    plt.ylabel('Corn Futures Price ($)', size=14, labelpad=30)
    plt.suptitle('Neural Network Predictions\nUsing Last Day\'s Price as Feature', size=20, alpha=.8)
    plt.savefig('../Figures/neuralnet_1day_lag.png')

if __name__ == '__main__':
    net_df = pd.read_csv('../full_database.csv')
    prices_arr = net_df['Inflation Adjusted Price'].values[::-1]
    scaler = MinMaxScaler(feature_range=(0,1))
    prices_arr = scaler.fit_transform(prices_arr)
    train_prices, test_prices = train_test_split(prices_arr)
    look_back=1
    train_x, train_y = create_lagged(train_prices, look_back)
    test_x, test_y = create_lagged(test_prices, look_back)
    model = make_net(train_x, train_y, 4, look_back)
    train_preds = make_preds(train_x, train_y, model)
    test_preds = make_preds(test_x, test_y, model)
    model.save('neuralnet_1day_lag.h5')
    plot_predictions()

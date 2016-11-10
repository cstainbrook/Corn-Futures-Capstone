from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.ext.CommissionReport import CommissionReport
from ib.ext.TickType import TickType as tt
from time import sleep, strftime
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime as dt, timedelta as td
import numpy as np

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

def update_dataframe():
    corn_df = pd.read_csv('corn_futures_prices.csv')
    todays_date = pd.datetime.today()
    last_date_entered = dt.strptime(corn_df['Date'][0], '%Y-%m-%d')
    num_days_missed = todays_date - last_date_entered
    for day in np.arange(1, num_days_missed.days):
    #intentionally leaving the current day off, so as to make this generalizable to any time during the day.
        if check_business_day(last_date_entered + td(days=day)):
            new_date = (last_date_entered + td(days=day))
            new_date = '{}-{}-{}'.format(new_date.year, new_date.month, new_date.day)
            new_price = prices[-(num_days_missed.days - day)]
            new_df = pd.DataFrame(np.array([[new_date, new_price]]), columns = ['Date', 'Settle'])
            corn_df = new_df.append(corn_df, ignore_index=True)[['Date', 'Settle']]
        else:
            pass

    corn_df.to_csv('corn_futures_prices.csv', index=False)

if __name__ == "__main__":

    prices = []

    conn = ibConnection(port=7497, clientId=110)
    conn.registerAll(all_handler)
    conn.unregister(all_handler, message.historicalData)
    conn.register(hist_data_handler, message.historicalData)
    conn.unregister(all_handler, 'TickPrice')
    conn.register(current_price_handler, 'TickPrice')
    conn.connect()

    month = dt.today().month
    year = dt.today().year
    # shortest_future_date =
    corn_future = PriceInformation('ZC', 'FUT', '20161214', 'ECBOT', 'USD')
    endtime = strftime('%Y%m%d %H:%M:%S')
    conn.reqHistoricalData(1,corn_future.contract,endtime,'14 D','1 day','TRADES',1,1)
    sleep(5)
    conn.disconnect()

    update_dataframe()

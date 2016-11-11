import pandas as pd
import numpy as np


def lag_six_months():
    prices = np.append(df['Settle'][127:], np.full(127, np.nan))
    supplies = np.append(df['Supply Level'][127:], np.full(127, np.nan))
    onis = np.append(df['ONI Index'][127:], np.full(127, np.nan))
    usd_index = np.append(df['USD Index'][127:], np.full(127, np.nan))
    soybeans = np.append(df['Soybean Prices'][127:], np.full(127, np.nan))
    crude = np.append(df['Oil Prices'][127:], np.full(127, np.nan))
    iowa_precip = np.append(df['Iowa 6 Month Precip'][127:], np.full(127, np.nan))
    iowa_temp = np.append(df['Iowa 6 Month Temp'][127:], np.full(127, np.nan))
    illinois_precip = np.append(df['Illinois 6 Month Precip'][127:], np.full(127, np.nan))
    illinois_temp = np.append(df['Illinois 6 Month Temp'][127:], np.full(127, np.nan))
    nebraska_precip = np.append(df['Nebraska 6 Month Precip'][127:], np.full(127, np.nan))
    nebraska_temp = np.append(df['Nebraska 6 Month Temp'][127:], np.full(127, np.nan))
    minnesota_precip = np.append(df['Minnesota 6 Month Precip'][127:], np.full(127, np.nan))
    minnesota_temp = np.append(df['Minnesota 6 Month Temp'][127:], np.full(127, np.nan))

    df['Price 6 Months Ago'] = prices
    df['Supply 6 Months Ago'] = supplies
    df['ONI Index 6 Months Ago'] = onis
    df['USD Index 6 Months Ago'] = usd_index
    df['Soybean Prices 6 Months Ago'] = soybeans
    df['Oil Prices 6 Months Ago'] = crude
    df['Iowa Precip 6 Months Ago'] = iowa_precip
    df['Iowa Temp 6 Months Ago'] = iowa_temp
    df['Illinois Precip 6 Months Ago'] = illinois_precip
    df['Illinois Temp 6 Months Ago'] = illinois_temp
    df['Nebraska Precip 6 Months Ago'] = nebraska_precip
    df['Nebraska Temp 6 Months Ago'] = nebraska_temp
    df['Minnesota Precip 6 Months Ago'] = minnesota_precip
    df['Minnesota Temp 6 Months Ago'] = minnesota_temp

    df.drop(['Supply Level', 'ONI Index', 'USD Index', 'Soybean Prices', 'Oil Prices', 'Iowa 6 Month Precip', 'Iowa 6 Month Temp', 'Illinois 6 Month Precip', 'Illinois 6 Month Temp', 'Nebraska 6 Month Precip', 'Nebraska 6 Month Temp', 'Minnesota 6 Month Precip', 'Minnesota 6 Month Temp'], axis=1, inplace=True)


if __name__ == '__main__':
    df = pd.read_csv('full_database.csv')
    df.drop([ 'Acreage Planted', 'Yield', 'Total Production (bushels)','CPI (2016=100)', 'Inflation Adjusted Price', 'daily change'], axis=1, inplace=True)
    df['Corn Syrup'] = df['Corn Syrup'].map(lambda x: float(x))
    df['E85'] = df['E85'].map(lambda x: float(x))
    lag_six_months()
    df.to_csv('six_month_database.csv')

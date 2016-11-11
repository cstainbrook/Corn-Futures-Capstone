import pandas as pd
import numpy as np


def lag_three_months():
    prices = np.append(df['Settle'][64:], np.full(64, np.nan))
    supplies = np.append(df['Supply Level'][64:], np.full(64, np.nan))
    onis = np.append(df['ONI Index'][64:], np.full(64, np.nan))
    usd_index = np.append(df['USD Index'][64:], np.full(64, np.nan))
    soybeans = np.append(df['Soybean Prices'][64:], np.full(64, np.nan))
    crude = np.append(df['Oil Prices'][64:], np.full(64, np.nan))
    iowa_precip = np.append(df['Iowa 6 Month Precip'][64:], np.full(64, np.nan))
    iowa_temp = np.append(df['Iowa 6 Month Temp'][64:], np.full(64, np.nan))
    illinois_precip = np.append(df['Illinois 6 Month Precip'][64:], np.full(64, np.nan))
    illinois_temp = np.append(df['Illinois 6 Month Temp'][64:], np.full(64, np.nan))
    nebraska_precip = np.append(df['Nebraska 6 Month Precip'][64:], np.full(64, np.nan))
    nebraska_temp = np.append(df['Nebraska 6 Month Temp'][64:], np.full(64, np.nan))
    minnesota_precip = np.append(df['Minnesota 6 Month Precip'][64:], np.full(64, np.nan))
    minnesota_temp = np.append(df['Minnesota 6 Month Temp'][64:], np.full(64, np.nan))

    df['Price 3 Months Ago'] = prices
    df['Supply 3 Months Ago'] = supplies
    df['ONI Index 3 Months Ago'] = onis
    df['USD Index 3 Months Ago'] = usd_index
    df['Soybean Prices 3 Months Ago'] = soybeans
    df['Oil Prices 3 Months Ago'] = crude
    df['Iowa Precip 3 Months Ago'] = iowa_precip
    df['Iowa Temp 3 Months Ago'] = iowa_temp
    df['Illinois Precip 3 Months Ago'] = illinois_precip
    df['Illinois Temp 3 Months Ago'] = illinois_temp
    df['Nebraska Precip 3 Months Ago'] = nebraska_precip
    df['Nebraska Temp 3 Months Ago'] = nebraska_temp
    df['Minnesota Precip 3 Months Ago'] = minnesota_precip
    df['Minnesota Temp 3 Months Ago'] = minnesota_temp

    df.drop(['Supply Level', 'ONI Index', 'USD Index', 'Soybean Prices', 'Oil Prices', 'Iowa 6 Month Precip', 'Iowa 6 Month Temp', 'Illinois 6 Month Precip', 'Illinois 6 Month Temp', 'Nebraska 6 Month Precip', 'Nebraska 6 Month Temp', 'Minnesota 6 Month Precip', 'Minnesota 6 Month Temp'], axis=1, inplace=True)


if __name__ == '__main__':
    df = pd.read_csv('full_database.csv')
    df.drop([ 'Acreage Planted', 'Yield', 'Total Production (bushels)','CPI (2016=100)', 'Inflation Adjusted Price', 'daily change'], axis=1, inplace=True)
    df['Corn Syrup'] = df['Corn Syrup'].map(lambda x: float(x))
    df['E85'] = df['E85'].map(lambda x: float(x))
    lag_three_months()
    df.to_csv('three_month_database.csv')

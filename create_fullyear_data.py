import pandas as pd
import numpy as np


def lag_one_year():
    prices = np.append(df['Settle'][252:], np.full(252, np.nan))
    supplies = np.append(df['Supply Level'][252:], np.full(252, np.nan))
    onis = np.append(df['ONI Index'][252:], np.full(252, np.nan))
    usd_index = np.append(df['USD Index'][252:], np.full(252, np.nan))
    soybeans = np.append(df['Soybean Prices'][252:], np.full(252, np.nan))
    crude = np.append(df['Oil Prices'][252:], np.full(252, np.nan))
    iowa_precip = np.append(df['Iowa 6 Month Precip'][252:], np.full(252, np.nan))
    iowa_temp = np.append(df['Iowa 6 Month Temp'][252:], np.full(252, np.nan))
    illinois_precip = np.append(df['Illinois 6 Month Precip'][252:], np.full(252, np.nan))
    illinois_temp = np.append(df['Illinois 6 Month Temp'][252:], np.full(252, np.nan))
    nebraska_precip = np.append(df['Nebraska 6 Month Precip'][252:], np.full(252, np.nan))
    nebraska_temp = np.append(df['Nebraska 6 Month Temp'][252:], np.full(252, np.nan))
    minnesota_precip = np.append(df['Minnesota 6 Month Precip'][252:], np.full(252, np.nan))
    minnesota_temp = np.append(df['Minnesota 6 Month Temp'][252:], np.full(252, np.nan))

    df['Price One Year Ago'] = prices
    df['Supply One Year Ago'] = supplies
    df['ONI Index One Year Ago'] = onis
    df['USD Index One Year Ago'] = usd_index
    df['Soybean Prices One Year Ago'] = soybeans
    df['Oil Prices One Year Ago'] = crude
    df['Iowa Precip One Year Ago'] = iowa_precip
    df['Iowa Temp One Year Ago'] = iowa_temp
    df['Illinois Precip One Year Ago'] = illinois_precip
    df['Illinois Temp One Year Ago'] = illinois_temp
    df['Nebraska Precip One Year Ago'] = nebraska_precip
    df['Nebraska Temp One Year Ago'] = nebraska_temp
    df['Minnesota Precip One Year Ago'] = minnesota_precip
    df['Minnesota Temp One Year Ago'] = minnesota_temp

    df.drop(['Supply Level', 'ONI Index', 'USD Index', 'Soybean Prices', 'Oil Prices', 'Iowa 6 Month Precip', 'Iowa 6 Month Temp', 'Illinois 6 Month Precip', 'Illinois 6 Month Temp', 'Nebraska 6 Month Precip', 'Nebraska 6 Month Temp', 'Minnesota 6 Month Precip', 'Minnesota 6 Month Temp'], axis=1, inplace=True)


if __name__ == '__main__':
    df = pd.read_csv('full_database.csv')
    df.drop([ 'Acreage Planted', 'Yield', 'Total Production (bushels)','CPI (2016=100)', 'Inflation Adjusted Price', 'daily change'], axis=1, inplace=True)
    df['Corn Syrup'] = df['Corn Syrup'].map(lambda x: float(x))
    df['E85'] = df['E85'].map(lambda x: float(x))
    lag_one_year()
    df.to_csv('one_year_database.csv')

import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup

def add_acreage():
    acreage_df = pd.read_csv('USDA Data/usda_acreage.csv')[['Year', 'Period', 'Value']]
    acreage_df = acreage_df[acreage_df['Period'] == 'YEAR']
    acreage_df.set_index('Year', inplace=True)
    main_df['Acreage Planted'] = main_df['Year'].map(lambda x: int(acreage_df['Value'][x].replace(',','')))

def add_yield():
    yield_df = pd.read_csv('USDA Data/usda_yield.csv')[['Year', 'Period', 'Value']]
    yield_df = yield_df[yield_df['Period'] == 'YEAR']
    yield_df.set_index('Year', inplace=True)
    main_df['Yield'] = main_df['Year'].map(lambda x: float(yield_df['Value'][x]))

def add_day_of_year():
    main_df['Day of Year'] = main_df['Date'].map(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime(datetime.strptime(x, '%Y-%m-%d').year - 1, 12, 31)).days)

def add_production():
    production_df = pd.read_csv('USDA Data/usda_production.csv')[['Year', 'Period', 'Value']]
    production_df = production_df[production_df['Period'] == 'YEAR']
    production_df.set_index('Year', inplace=True)
    main_df['Total Production (bushels)'] = main_df['Year'].map(lambda x: float(production_df['Value'][x].replace(',', '')))

def add_supply():
    '''Linearly smoothing supply levels between usda reports.'''

    supply_df = pd.read_csv('USDA DATA/usda_supply_levels.csv')
    supply_df['year'] = supply_df['Unnamed: 0']
    supply_df['year'].fillna(method='ffill', inplace=True)
    supply_df.dropna(subset=['Unnamed: 1'], inplace=True)
    supply_df['month'] = supply_df['Unnamed: 1'].map(lambda x: x.split()[1].split('-')[0])
    current_year_mask = (supply_df['month'] == 'Sep') | (supply_df['month'] == 'Dec')
    supply_df.year[current_year_mask] = supply_df.year[current_year_mask].map(lambda x: x.split('/')[0])
    supply_df.year[~current_year_mask] = supply_df.year[~current_year_mask].map(lambda x: ''.join(list(x)[0:2] + list(x)[-2:]))
    month_dict = {'Sep': '09', 'Dec': '12', 'Mar': '03', 'Jun': '06'}
    supply_df['month'] = supply_df['month'].map(lambda x: month_dict[x])
    supply_df['date'] = supply_df['year'] + '-' + supply_df['month'] + '-01'
    supply_df.set_index('date', inplace=True)
    supply_df.drop(['year', 'month'], axis=1, inplace=True)
    supply_df.drop_duplicates(subset='Beginning stocks', keep='first', inplace=True)

    main_df['Supply Level'] = main_df['Date'].map(lambda x: float(''.join(supply_df['Beginning stocks'][x].split(','))) if x in supply_df['Beginning stocks'] else np.NaN)

    #Interpolating all data points between readings.  The first 40 points are set to the last reading.  All points before the first reading are left as nans.
    main_df['Supply Level'][40:10308] = main_df['Supply Level'][40:10308].interpolate()
    main_df['Supply Level'][0:40] = main_df['Supply Level'][40]

def add_oni():
    nino_df = pd.read_csv('oni.csv')
    nino_df['Date'] = nino_df['PeriodNum'] + '-01'
    nino_df.set_index('Date', inplace=True)
    main_df['ONI Index'] = main_df['Date'].map(lambda x: nino_df['Value'][x] if x in nino_df['Value'] else np.NaN)
    main_df['ONI Index'].fillna(method='backfill', inplace=True)

def add_demand_dummies():
    main_df['Corn Syrup'] = main_df['Date'].map(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d') > datetime(1969, 12, 31) else 0)
    main_df['E85'] = main_df['Date'].map(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d') > datetime(1999, 12, 31) else 0)

def inflation_adjust():
    inflation_df = pd.read_csv('cpi.csv')
    current_year_avg = inflation_df['CPIAUCSL'][828:].mean()
    inflation_df['CPI (2016=100)'] = inflation_df['CPIAUCSL'].map(lambda x: (x/current_year_avg)*100)
    inflation_df.drop('CPIAUCSL', axis=1)
    inflation_df.set_index('DATE', inplace=True)
    main_df['CPI (2016=100)'] = main_df['Date'].map(lambda x: inflation_df['CPI (2016=100)'][x] if x in inflation_df['CPI (2016=100)'] else np.NAN)
    main_df['CPI (2016=100)'].fillna(method='backfill', inplace=True)
    main_df['Inflation Adjusted Price'] = main_df['Settle'] / main_df['CPI (2016=100)'] * 100

def add_daily_change():
    main_df['daily change'] = main_df['Inflation Adjusted Price'].diff() * -1
    main_df.drop(0, axis=0, inplace=True)
    main_df.drop([10901, 10902], axis=0, inplace=True)

def add_usd_index():
    usd_df = pd.read_csv('usd_index.csv', usecols=['Date','Settle'])
    usd_df.set_index('Date', inplace=True)
    main_df['USD Index'] = main_df['Date'].map(lambda x: usd_df['Settle'][x] if x in usd_df['Settle'] else np.NAN)

def add_soybean_prices():
    soybean_df = pd.read_csv('soybean_futures_prices.csv', usecols=['Date','Settle'])
    soybean_df.set_index('Date', inplace=True)
    main_df['Soybean Prices'] = main_df['Date'].map(lambda x: soybean_df['Settle'][x] if x in soybean_df['Settle'] else np.NAN)

def add_oil_prices():
    oil_df = pd.read_csv('oil_futures_prices.csv', usecols=['Date','Settle'])
    oil_df.set_index('Date', inplace=True)
    main_df['Oil Prices'] = main_df['Date'].map(lambda x: oil_df['Settle'][x] if x in oil_df['Settle'] else np.NAN)

def add_weather_data():
    iowa_df = pd.read_csv('Weather Data/iowa_weather_full.csv')
    iowa_df.drop_duplicates('Date', inplace=True)
    iowa_df.set_index('Date', inplace=True)
    iowa_df['Six Month Precip'] = pd.rolling_mean(iowa_df['Precipitation'], 182)
    iowa_df['Six Month Precip'] = iowa_df['Six Month Precip'].map(lambda x: round(x, 2))
    iowa_df['Six Month Temp'] = pd.rolling_mean(iowa_df['Maximum Temp'], 182)
    iowa_df['Six Month Temp'] = iowa_df['Six Month Temp'].map(lambda x: round(x, 2))
    main_df['Iowa 6 Month Precip'] = main_df['Date'].map(lambda x: iowa_df['Six Month Precip'][x] if x in iowa_df['Six Month Precip'] else np.NAN)
    main_df['Iowa 6 Month Temp'] = main_df['Date'].map(lambda x: iowa_df['Six Month Temp'][x] if x in iowa_df['Six Month Temp'] else np.NAN)

    illinois_df = pd.read_csv('Weather Data/illinois_weather_full.csv')
    illinois_df.drop_duplicates('Date', inplace=True)
    illinois_df.set_index('Date', inplace=True)
    illinois_df['Six Month Precip'] = pd.rolling_mean(illinois_df['Precipitation'], 182)
    illinois_df['Six Month Precip'] = illinois_df['Six Month Precip'].map(lambda x: round(x, 2))
    illinois_df['Six Month Temp'] = pd.rolling_mean(illinois_df['Maximum Temp'], 182)
    illinois_df['Six Month Temp'] = illinois_df['Six Month Temp'].map(lambda x: round(x, 2))
    main_df['Illinois 6 Month Precip'] = main_df['Date'].map(lambda x: illinois_df['Six Month Precip'][x] if x in illinois_df['Six Month Precip'] else np.NAN)
    main_df['Illinois 6 Month Temp'] = main_df['Date'].map(lambda x: illinois_df['Six Month Temp'][x] if x in illinois_df['Six Month Temp'] else np.NAN)

    nebraska_df = pd.read_csv('Weather Data/nebraska_weather_full.csv')
    nebraska_df.drop_duplicates('Date', inplace=True)
    nebraska_df.set_index('Date', inplace=True)
    nebraska_df['Six Month Precip'] = pd.rolling_mean(nebraska_df['Precipitation'], 182)
    nebraska_df['Six Month Precip'] = nebraska_df['Six Month Precip'].map(lambda x: round(x, 2))
    nebraska_df['Six Month Temp'] = pd.rolling_mean(nebraska_df['Maximum Temp'], 182)
    nebraska_df['Six Month Temp'] = nebraska_df['Six Month Temp'].map(lambda x: round(x, 2))
    main_df['Nebraska 6 Month Precip'] = main_df['Date'].map(lambda x: nebraska_df['Six Month Precip'][x] if x in nebraska_df['Six Month Precip'] else np.NAN)
    main_df['Nebraska 6 Month Temp'] = main_df['Date'].map(lambda x: nebraska_df['Six Month Temp'][x] if x in nebraska_df['Six Month Temp'] else np.NAN)

    minnesota_df = pd.read_csv('Weather Data/minnesota_weather_full.csv')
    minnesota_df.drop_duplicates('Date', inplace=True)
    minnesota_df.set_index('Date', inplace=True)
    minnesota_df['Six Month Precip'] = pd.rolling_mean(minnesota_df['Precipitation'], 182)
    minnesota_df['Six Month Precip'] = minnesota_df['Six Month Precip'].map(lambda x: round(x, 2))
    minnesota_df['Six Month Temp'] = pd.rolling_mean(minnesota_df['Maximum Temp'], 182)
    minnesota_df['Six Month Temp'] = minnesota_df['Six Month Temp'].map(lambda x: round(x, 2))
    main_df['Minnesota 6 Month Precip'] = main_df['Date'].map(lambda x: minnesota_df['Six Month Precip'][x] if x in minnesota_df['Six Month Precip'] else np.NAN)
    main_df['Minnesota 6 Month Temp'] = main_df['Date'].map(lambda x: minnesota_df['Six Month Temp'][x] if x in minnesota_df['Six Month Temp'] else np.NAN)

def make_full_database():
    add_day_of_year()
    add_acreage()
    add_yield()
    add_production()
    add_supply()
    add_oni()
    add_demand_dummies()
    inflation_adjust()
    add_daily_change()
    add_usd_index()
    add_soybean_prices()
    add_oil_prices()
    add_weather_data()

    main_df.drop('Year', axis=1, inplace=True)
    main_df['Date'] = main_df['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    main_df.set_index('Date', inplace=True)


if __name__ == '__main__':
    main_df = pd.read_csv('corn_futures_prices.csv')
    main_df['Year'] = main_df['Date'].map(lambda x: int(x.split('-')[0]))
    make_full_database()

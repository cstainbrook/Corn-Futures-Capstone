import requests
import os
import json
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime, timedelta

def make_noaa_request(date, fips):
    '''Makes a request to the NOAA API and appends one day of data to the larger dataframe.

    INPUT: string, integer
    OUTPUT: Pandas Dataframe
    '''

    url = 'http://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&datatypeid=PRCP&datatypeid=TMAX&locationid=FIPS:{}&startdate={}&enddate={}&limit=1000'.format(fips, date, date)

    response = requests.get(url, headers={'token': token})
    data = response.json()

    df = json_normalize(data['results'])
    df = df.groupby(['datatype']).mean()
    new_df = pd.DataFrame([[date, fips, df['value']['PRCP'], df['value']['TMAX']]], columns=['Date', 'State_FIPS', 'Precipitation', 'Maximum Temp'])
    weather_df_new = weather_df.append(new_df, ignore_index=True)
    return weather_df_new

if __name__ == '__main__':

    weather_df = pd.DataFrame()
    fips = 18
    token = os.environ['NOAA_API_0']

    date = datetime(1959, 7, 8)
    end_date = datetime(2016, 10, 30)
    while date < end_date:
        str_date = str(date).split()[0]
        print str_date
        weather_df = make_noaa_request(str_date, fips)
        date = date + timedelta(days=1)

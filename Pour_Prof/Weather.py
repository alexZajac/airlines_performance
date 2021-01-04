import pandas as pd
from meteostat import Stations, Daily
from tqdm import tqdm
from geopy.geocoders import Nominatim
from functools import partial
import ast
import datetime

path = 'data/'
features = ['TAVG', 'TMAX', 'PRCP', 'SNOW', 'WSPD', 'TSUN']
tqdm.pandas()
geolocator = Nominatim(user_agent="application_name", timeout=6000)

def get_city_statistics(row, start_date, end_date, features, agg_freq="1M"):
    """Fetches all weather given stats for a longitude and latitude"""
    # get coordinates
    latitude, longitude = row["POINT"]
    # fetch closest weather station
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    stations = stations.inventory("daily", (start_date, end_date))
    station = stations.fetch(1)
    # get daily data
    data = Daily(station, start_date, end_date)
    data = data.normalize().aggregate(freq=agg_freq).fetch()
    # check if there is data for this station
    if data.shape[0] < 1:
        return None
    # select required features
    data = data[features]
    data.fillna(0., inplace=True)
    return data

##Get lat and long fro each destination

def lat_long_destinations(df) : 
    unique_destinations = pd.DataFrame(df["DEST_CITY_NAME"].unique(), columns=["DEST_CITY_NAME"])
    unique_destinations["LOCATION"] = unique_destinations["DEST_CITY_NAME"].progress_apply(geolocator.geocode)
    unique_destinations["POINT"] = unique_destinations["LOCATION"].apply(lambda loc: tuple((loc.point[0], loc.point[1])) if loc else (0., 0.))
    del unique_destinations["LOCATION"]
    return unique_destinations

# get the weather information
def get_weather_df(unique_destinations):
    unique_destinations["POINT"] = unique_destinations["POINT"].apply(lambda x: ast.literal_eval(x))
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    lower_features = [f.lower() for f in features]

    unique_destinations["WEATHER_STATS"] = unique_destinations.progress_apply(
        partial(get_city_statistics, start_date=start_date, end_date=end_date, features=lower_features),
        axis=1
    )
    unique_destinations["DATE"] = unique_destinations.apply(lambda row: pd.date_range(start_date, end_date, freq="1M"), axis=1)
    unique_destinations = unique_destinations.explode("DATE").reset_index()#drop = True ?
    unique_destinations[["TAVG", "TMAX", "PRCP", "SNOW", "WSPD", "TSUN"]] = unique_destinations.progress_apply(
        lambda row: row["WEATHER_STATS"].iloc[row.name % len(row["WEATHER_STATS"].index)], 
        axis=1
    )
    del unique_destinations["WEATHER_STATS"]

df = pd.read_csv(path + 'features.csv')
unique_destinations =  lat_long_destinations(df)
weather_df = get_weather_df(unique_destinations)
weather_df.to_csv("weather_dataset.csv")
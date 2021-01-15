import pandas as pd
from meteostat import Stations, Daily
from tqdm import tqdm
from geopy.geocoders import Nominatim
from functools import partial
import ast
import datetime
from pathlib import Path

path = Path("../data")
features = ['TAVG', 'TMAX', 'PRCP', 'SNOW', 'WSPD', 'TSUN']
tqdm.pandas()
geolocator = Nominatim(user_agent="city_coordinator", timeout=6000)


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


def get_lat_long_destinations(df):
    """Get lat and long for each destination"""
    unique_destinations = pd.DataFrame(
        # Top1 destination n'est qu' un exmpele de destination pour sortir les donnee
        df["DEST_CITY_NAME"].unique(), columns=["DEST_CITY_NAME"]
    )
    unique_destinations["LOCATION"] = unique_destinations["DEST_CITY_NAME"].progress_apply(
        geolocator.geocode
    )
    unique_destinations["POINT"] = unique_destinations["LOCATION"].apply(
        lambda loc: tuple((loc.point[0], loc.point[1])) if loc else (0., 0.)
    )
    del unique_destinations["LOCATION"]
    return unique_destinations


def extract_weather_statistics(df, start_date, end_date):
    """Calls the meteostat API to collect weather statistics"""
    lower_features = [f.lower() for f in features]
    df["WEATHER_STATS"] = df.progress_apply(
        partial(
            get_city_statistics,
            start_date=start_date,
            end_date=end_date,
            features=lower_features
        ),
        axis=1
    )
    return df


def get_weather_df(unique_destinations):
    """Applies the whole pipeline to get weather features"""
    start_date = datetime.datetime(2013, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)
    unique_destinations = extract_weather_statistics(
        unique_destinations, start_date, end_date
    )

    # expand dimensions to one motnh for each airlines/corrdinate pair
    unique_destinations["DATE"] = unique_destinations.apply(
        lambda row: pd.date_range(start_date, end_date, freq="1M"), axis=1)
    unique_destinations = unique_destinations.explode(
        "DATE").reset_index(drop=True)
    unique_destinations[features] = unique_destinations.progress_apply(
        lambda row: row["WEATHER_STATS"].iloc[
            row.name % len(row["WEATHER_STATS"].index)
        ],
        axis=1
    )
    del unique_destinations["WEATHER_STATS"]
    return unique_destinations


if __name__ == "__main__":
    df = pd.read_csv(path / 'root.csv')
    unique_destinations = get_lat_long_destinations(df)
    weather_df = get_weather_df(unique_destinations)
    weather_df.to_csv("weather_dataset.csv")

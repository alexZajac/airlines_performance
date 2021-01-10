# Explanations on the process for getting the final datasets

## Transats Dataset: Choice, collection and feature engineering ✈️

### Choice of dataset

We first chose a baseline place for building a dataset given our problem, and after a lot of investigations, we chose to go for an US-based dataset since they provide an enormous amount of databases, tables and features to chose from. Our logic was first breadth-oriented, since there exists a lot of different websites that source aviation data, but we finally settled on the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/) because it's the source for a lot of others datasets.

This dataset is the preeminent source of statistics on aviation, multimodal freight activity and transportation economics in the United States.
It is a very popular and used source of informations for both political, commercial and public use.
It is part of the US Department of Transportation (DOT).

We chose to use the dataset (**T-100 Domestic Segment**) containing detailed informations on the flights for US Carriers only like `PASSENGERS, SEATS, PAYLOAD, DESTINATION_CITY, AIRCRAFT_TYPE, ...`
We've transformed this dataset so it was indexed by `DATE` (month) and by `UNIQUE_CARRIER_NAME` over some recent years. We gathered the informations by month and by companies and enriched the dataset with statistical features, temperatures, top destinations, etc... from there. Our goal was to extract as much pertinent features as possible to make a relevant dataset that would be interesting to work on.

### Data collection and feature engineering

Since part of the data we wanted on the website wasn't directly available for all the given years (2014-2020), we put up a simple script that copied the network request from the transtats website in Node.JS (chrome only let us get the request via curL or Node.JS fetch), and from there we saw that we could modify the fields and filter parameters from the SQL query the site was running, to get all the fields we wanted. It's the `fetch_from_transats.js` script.

The notebook `data_engineering.ipynb` presents a subset of the feature engineering and cleaning work we have done on the original dataset.

## Weather dataset: Choice of dataset and extraction 🌤️

### Choice of dataset

We made the choice to enrich our base dataset with meteorological features, since we thought they could have a great influence on the load factor for an airline/carrier.

On the one hand, snow, rain and crosswinds mean that air traffic controllers have to increase the gap between planes that are landing, reducing the number of aircraft that an airport can manage. The same weather can make it slower and more difficult for the planes to taxi between runway and terminal building. On the other hand, the average temperature and sunshine level on the arrival destination can influence the choice of passengers for a given carrier.

A lot of APIs provide daily or weekly live forecast for weather data but few provide an historical record for free. We finally found the [meteostat website](https://dev.meteostat.net/) that provides JSON APIs, a python library and even a web app to directly visualize weather data accross the entire globe, for free (long live open-source)!

### Data collection and weather statistics

Because the weather APIs of meteostats have a nice built-in function to determine which weather station is the closest to the city where we want to query the weather data for, we used the `geopy` library to get all the coordinates of the possibles destinations for the flights of our dataset. The weather data is then retrieved from these given weather stations.

We chose the features in line with our problem: `tavg, tmax, tsun, snow, wspd and prcp` (as given [here](https://dev.meteostat.net/python/daily.html#data-structure)). Since our data has a monthly granularity, we directly aggregated the data with the python API meteostats provides, and normalized the data to account for temporal outage of some of the sensors from the weather stations.

You can test a full run of data extraction with the `weather_data.py` script.

## Twitter tweets: Collection with web scraping and sentiment analysis 🐦

### Choice of dataset

It's not uncommon to include Twitter data in machine learning challenges. Since we are forecasting a load factor, which is a proxy for the performance of an airline carrier, we thought that incorporating tweets mentionning the chosen airlines could give more breadth to the features of our dataset. More precisely, by performing sentiment analysis on the tweets of a given airline, one could get a sense and new feature to test for the prediction model.

### Data collection

There exist an API for getting tweets from Twitter, so we signed up for a developer account, but the free version only allows to get historical data up to 7 days. So we switched to libraries performing web scraping to get tweets, and there are a lot of them (probably too much), and after a lot of trail and error we settled on [scweet](https://github.com/Altimis/Scweet), forked the repository and made a custom script for our requirements. Since scraping 6 years of tweets for 23 carriers can be very long, our was parallelized to launch one headless chrome instance per core, with an interval of 30 days of tweets to scrape as we found this to give a good trade-off.

### Sentiment analysis

We then set-up another script to convert all these tweets Dataframes into dataframes with sentiments probabilities (negative, neutral or positive), and for this we used the amazing [Hugging Face Transformers library](https://huggingface.co/) and more particularly the [Roberta model trained on tweets](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) to extract 3 sentiment probabilities for each tweets. As some carriers are lot less present in the Twitter feed, there are some discrepencies of dataset size and we found ways to handle them.

One particular point to note here is that we didn't let the students get the full sentiment-lablized tweets dataset, but showed them our results with it, so that they see a real value in dealing with twitter data for the predictive task.

You can test a full run of data extraction with the `twitter_data.py` script. To test this you will need to have a chrome driver instance in the folder `./drivers`.

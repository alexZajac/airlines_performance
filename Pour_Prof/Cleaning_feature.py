import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

path = 'data/'
# Read the CSV file, delete unnamed columns, creates load factor, keeps only number of seats > 10, creates DATE column
def read_n_clean(name):
  df = pd.read_csv(name)
  # Drop Unnamed Columns
  df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis = 1, inplace = True)
  df['DATE'] =  pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
  # We pick only data from 2014 to 2019
  date_ref1 = datetime.datetime(2014, 1, 1)
  date_ref2 = datetime.datetime(2020, 1, 1)
  df = df[(df['DATE'] > date_ref1) & (df['DATE'] < date_ref2)]
  # We pick only flights with available seats > 10
  df = df[df["SEATS"] > 10].reset_index(drop=True)  
  # We pick only flights with non-null distance travelled 
  df = df[df["DISTANCE"] > 0].reset_index(drop=True) 
  # We dropped the Hageland Airline data -> the company has shut down since 2008 and has very low amounts of tweets
  df = df[df['UNIQUE_CARRIER'] != "H6"]
  # Load-Factor Creation
  df["RPM"] = df["PASSENGERS"] * df["DISTANCE"]
  df["ASM"] = df["SEATS"] * df["DISTANCE"]
  return df



##Choice of Ariline
def Choice_Carrier(df):
    count_flights = df.groupby(['UNIQUE_CARRIER_NAME', 'DATE'])['PASSENGERS'].count().reset_index(name = 'NB_FLIGHTS')
    count_flights_f = count_flights.groupby(['UNIQUE_CARRIER_NAME'])['NB_FLIGHTS'].mean().reset_index(name = 'MOY_NB_FLIGHTS').sort_values('MOY_NB_FLIGHTS', ascending = False)
    # TOT Passengers
    tot_pass = count_flights_f.merge(df.groupby(['UNIQUE_CARRIER_NAME'])['PASSENGERS'].sum().reset_index(name = 'TOT_PASS_FLIGHTS').reset_index(drop=True), on= ['UNIQUE_CARRIER_NAME']).sort_values('TOT_PASS_FLIGHTS', ascending = False)
    # NB Flights
    tot_nb_flights = tot_pass.merge(df.groupby(['UNIQUE_CARRIER_NAME'])['PASSENGERS'].count().reset_index(name = 'TOT_NB_FLIGHTS').reset_index(drop=True), on=['UNIQUE_CARRIER_NAME']).sort_values('TOT_NB_FLIGHTS', ascending = False)

    copy = tot_nb_flights.copy()
    scaler = StandardScaler()
    copy.iloc[:, 1:] = scaler.fit_transform(copy.iloc[:, 1:])
    copy['TOTAL'] = copy['MOY_NB_FLIGHTS'] + copy['TOT_PASS_FLIGHTS'] + copy['TOT_NB_FLIGHTS']
    copy.sort_values('TOTAL', ascending=False, inplace=True)


    copy2 = copy.iloc[:22, :].reset_index(drop=True)
    carriers_f = copy2['UNIQUE_CARRIER_NAME'].to_numpy().tolist()
    return carriers_f

df = read_n_clean( path + "T_100_Domestic_Segment_All_Years_Extended.csv")
carriers_f = Choice_Carrier(df)
df_2 = df[df['UNIQUE_CARRIER_NAME'].isin(carriers_f)].sort_values('DISTANCE')
df_2.to_csv(path + 'features.csv')

############################################################
############################################################

def extract_monthly_load_factor(data):
  tmp = data.groupby(['UNIQUE_CARRIER_NAME', 'DATE'])["RPM"].sum().reset_index(name="RPM_SUM")
  tmp2 = data.groupby(['UNIQUE_CARRIER_NAME', 'DATE'])["ASM"].sum().reset_index(name="ASM_SUM")
  merged = tmp.merge(tmp2, on=['UNIQUE_CARRIER_NAME', 'DATE'])
  merged["LOAD_FACTOR"] = merged["RPM_SUM"]/merged["ASM_SUM"]
  merged.drop(columns=['RPM_SUM', 'ASM_SUM'], inplace=True)
  return merged

load_factor_data = extract_monthly_load_factor(df_2)
load_factor_data.to_csv(path + 'target.csv')
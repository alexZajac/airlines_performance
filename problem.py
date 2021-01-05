import pandas as pd

def get_data(trainOrtest, path="."):
    X_df = pd.read_csv(trainOrtest + 'X.csv' )
    y = pd.read_csv(trainOrtest + 'y.csv' )
    return  X_df, y

import pandas as pd
import os

def get_data(str):
    absolute_path = os.path.abspath('')
    if str == "train":
        data = pd.read_csv( os.path.join(absolute_path, "house-prices-data/train.csv"))
    elif str == "test":
        data = pd.read_csv(os.path.join(absolute_path, "house-prices-data/test.csv"))
    else: 
        data = "Please insert train or test"
    return data 
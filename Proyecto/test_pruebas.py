from Modules import load_data
from Modules import data_preprocessing
import pandas as pd
from pandas.testing import assert_frame_equal
data = pd.read_csv("house-prices-data/test.csv")



def test_get_data():
    assert_frame_equal(load_data.get_data('test'), data)
    
def test_new_col_creation():
    assert( data_preprocessing.new_col_creation(data,data,'-','var1', 'var2', 'var3') == "Please input + or * ")
    


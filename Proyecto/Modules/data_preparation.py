import pandas as pd

def fill_null_values(df,col,new_val):
    for i in col:
        df[i].fillna(new_val, inplace=True)

        
def fill_all_missing_values(data):
    for col in data.columns:
        if((data[col].dtype == 'float64') or (data[col].dtype == 'int64')):
            data[col].fillna(data[col].mean(), inplace=True)
        else:
             data[col].fillna(data[col].mode()[0], inplace=True)
                
def drop_columns(df,drop_col):
    df.drop(drop_col, axis=1, inplace=True)
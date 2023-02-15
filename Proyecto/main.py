"""
This is the main script of the House Prices
- Advanced Regression Techniques document
This scripts import libraries and created modules.
The script has the following logic:
1) Load the data
2) Calculate EDA
3) Data preparation
4) Data processing
5) Model the data
"""
# Importing Libraries
import warnings
import logging
import argparse

# Imported modules
from Modules import load_data
from Modules import get_eda
from Modules import data_preparation
from Modules import data_preprocessing
from Modules import train_model

#0 Create logger for the script 
logging.basicConfig(filename='logs/logs.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

# 1) Load data
try:
    parser = argparse.ArgumentParser(
        prog="Insert the word ""Id", 
        usage="Please insert the desired column for testing, suggested: 'Id'",
        description="User should decide which column they want for testinhg",
        epilog="Ids")
    parser.add_argument('column', type=str)
    args = parser.parse_args()
    train_data = load_data.get_data('train')
    test_data = load_data.get_data('test')
    test_ids = test_data[args.column]
    print("Shape:", train_data.shape)
    print("Duplicated data :", train_data.duplicated().sum())
except FileNotFoundError:
    print("El archivo no se encontró, verificar el path")
    logging.error("Path erroneo")
except: 
    print(" ERRROR : To run the code please write python main.py 'Id'")
    logging.error("Parametro erroneo del usuario")
    
    

# 2)EDA
try: 
    get_eda.get_heatmap(train_data)
    get_eda.get_count_plot(df=train_data, var='SaleCondition')
    get_eda.get_histplot(df=train_data, var='SaleType')
    get_eda.get_violinplot(df=train_data, var1='HouseStyle', var2='SalePrice')
    get_eda.get_scatterplot(df=train_data, var1="Foundation", var2="SalePrice")
except: 
    logging.error("Something went wrong in the exploratory data analysis ")
    

# 3) Data preparation
try: 
    col = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
    data_preparation.fill_null_values(df=train_data, col=col, new_val="No")
    data_preparation.fill_all_missing_values(train_data)
    data_preparation.fill_all_missing_values(test_data)
    drop_col = [
            'Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold',
            'YrSold', 'MSSubClass', 'GarageType', 'GarageArea',
            'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
            'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF',
            'Condition1', 'Condition2', 'Heating', 'Exterior1st',
            'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour',
            'LotConfig', 'Functional', 'BsmtFinSF1', 'BsmtFinSF2',
            'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond',
            'OverallCond'
           ]
    data_preparation.drop_columns(train_data, drop_col)
    data_preparation.drop_columns(test_data, drop_col)
except :
    logging.error("Something went wrong with the data preparation")
    

# 4) Data preprocessing
try:
    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['BsmtQual', 'BsmtCond'],
                                ['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['ExterQual', 'ExterCond', 'KitchenQual'],
                                ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

    data_preprocessing.encode_ordinal(
                                train_data, test_data,
                                ['PavedDrive'],
                                ['N', 'P', 'Y'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['Electrical'],
                                ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'])

    data_preprocessing.encode_ordinal(
                                train_data, test_data,
                                ['BsmtFinType1',
                                 'BsmtFinType2'],
                                ['No', 'Unf',
                                 'LwQ', 'Rec',
                                 'BLQ', 'ALQ', 'GLQ'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['Utilities'],
                                ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['MSZoning'],
                                ['C (all)', 'RH', 'RM', 'RL', 'FV'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['Foundation'],
                                ['Slab', 'BrkTil', 'Stone',
                                 'CBlock', 'Wood', 'PConc'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['Neighborhood'],
                                [
                                    'MeadowV', 'IDOTRR', 'BrDale', 'Edwards',
                                    'BrkSide', 'OldTown', 'NAmes', 'Sawyer',
                                    'Mitchel', 'NPkVill', 'SWISU', 'Blueste',
                                    'SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn',
                                    'ClearCr', 'Crawfor', 'CollgCr', 'Veenker',
                                    'Timber', 'Somerst', 'NoRidge', 'StoneBr',
                                    'NridgHt'
                                ])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['MasVnrType'],
                                ['None', 'BrkCmn', 'BrkFace', 'Stone'])

    data_preprocessing.encode_ordinal(
                                train_data, test_data, ['SaleCondition'],
                                ['AdjLand', 'Abnorml', 'Alloca',
                                 'Family', 'Normal', 'Partial'])

    data_preprocessing.encode_ordinal(
                                train_data, test_data, ['RoofStyle'],
                                ['Gambrel', 'Gable', 'Hip',
                                 'Mansard', 'Flat', 'Shed'])

    data_preprocessing.encode_ordinal(
                                train_data,
                                test_data,
                                ['RoofMatl'],
                                ['ClyTile', 'CompShg', 'Roll', 'Metal',
                                 'Tar&Grv', 'Membran', 'WdShake', 'WdShngl'])

    Level_col = ['Street', 'BldgType', 'SaleType', 'CentralAir']

    data_preprocessing.encode_catagorical_columns(
                                            train_data, test_data, Level_col)
    data_preprocessing.new_col_creation(
                                  train_data,
                                  test_data,
                                  '*',
                                  'BsmtRating',
                                  'BsmtCond',
                                  'BsmtQual')

    data_preprocessing.new_col_creation(
                                  train_data,
                                  test_data,
                                  '*',
                                  'ExterRating',
                                  'ExterCond',
                                  'ExterQual')

    data_preprocessing.new_col_creation(
                                  train_data,
                                  test_data,
                                  '*',
                                  'BsmtFinTypeRating',
                                  'BsmtFinType1',
                                  'BsmtFinType2')

    data_preprocessing.new_col_creation(
                                  train_data,
                                  test_data,
                                  '+',
                                  'BsmtBath',
                                  'BsmtFullBath',
                                  'BsmtHalfBath')

    data_preprocessing.new_col_creation(
                                    train_data,
                                    test_data,
                                    '+',
                                    'Bath',
                                    'FullBath',
                                    'HalfBath')

    data_preprocessing.new_col_creation(
                                    train_data,
                                    test_data,
                                    '+',
                                    'PorchArea',
                                    'OpenPorchSF',
                                    'EnclosedPorch',
                                    '3SsnPorch',
                                    'ScreenPorch')
except:
    logging.error("Something went wrong with the data preprocesing module")
    
try: 
    drop_col = ['OverallQual', 'ExterCond', 'ExterQual', 'BsmtCond', 'BsmtQual',
            'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'OpenPorchSF',
            'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'BsmtFullBath',
            'BsmtHalfBath', 'FullBath', 'HalfBath']

    data_preparation.drop_columns(train_data, drop_col)
    data_preparation.drop_columns(test_data, drop_col)
    print(train_data.shape)
except: 
    logging.error("Something went wrong with the data preprocesing module")

# 5 Model
try: 
    modelo = train_model.model(250)
    train_model.score(train_data, "SalePrice", modelo)
    warnings.filterwarnings('ignore')
    train_model.model_prediction(modelo, test_data, test_ids).sample(10)
except: 
    logging.error("Something went wrong with the data preprocesing module")

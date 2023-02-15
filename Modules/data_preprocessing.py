from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

def encode_ordinal(df_train,df_test,var,cat):
    for i in var: 
        OE = OrdinalEncoder(categories=[cat])
        df_train[i] =OE.fit_transform(df_train[[i]])
        df_test[i] =OE.transform(df_test[[i]])
        
def encode_catagorical_columns(train, test, Level_col):
    for col in Level_col:
        train[col] = encoder.fit_transform(train[col])
        test[col]  = encoder.transform(test[col])

def new_col_creation(df_train,df_test,math_symbol, new, input1, input2, input3=0,input4=0): 
    if math_symbol =='+': 
        df_train[new]= df_train[input1]+ df_train[input2]
    elif math_symbol == '+' and input3 !=0 and input4 !=0: 
        df_test[new] = df_test[input1]+df_test[input2]+df_test[input3]+df_test[input4]
    elif math_symbol == '*' : 
        df_test[new] = df_test[input1]*df_test[input2]
    else:
        return "Please input + or * "

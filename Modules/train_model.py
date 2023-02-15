import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def model(max_leaf):
    return RandomForestRegressor(max_leaf_nodes= max_leaf,)
    
def score(train_data,target_var, model):
    y = train_data[target_var]
    X = train_data.drop([target_var], axis=1)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10)
    return print(score.mean())
    
def model_prediction(model, test_data,test_ids):
    price = model.predict(test_data)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": price
    })
    submission.to_csv("submission.csv", index=False)
    return submission
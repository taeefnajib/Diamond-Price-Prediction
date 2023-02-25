# import all dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass 
class Hyperparameters(object):
    filepath: str = "train.csv",
    test_size: float = 0.15,
    random_state: int = 786,
    num_leaves: int = 10,
    learnnig_rate: float = 0.05,


hp = Hyperparameters()

cat_cols = ["cut","color","clarity"]

# collect data
def get_data(filepath):
    return pd.read_csv("train.csv")

# preprocess data
def preprocess_data(df, cat_cols):
    df.drop(["id"], axis=1, inplace=True)
    df = pd.get_dummies(data=df, columns=cat_cols)
    return df

# split dataset
def split_data(df, test_size, random_state):
    X = df.drop(["price"], axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=66)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=66)
    return X_train, X_val, y_train, y_val 

# laoding data
def load_data(X_train, y_train, X_val, y_val):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    return lgb_train, lgb_eval

# defining parameters 
def get_params(num_leaves, learning_rate):
    params = {
        'task': 'train', 
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': num_leaves,
        'learnnig_rate': learning_rate,
        'metric': {'l2','l1'},
        'verbose': -1
    }
    return params

# train model
def train_model(params, lgb_train, lgb_eval):
    return lgb.train(params, train_set=lgb_train, valid_sets=lgb_eval)


# run workflow
def run_wf(hp: Hyperparameters)->lgb:
    df = get_data(filepath=hp.filepath)
    df = preprocess_data(df=df, cat_cols=cat_cols)
    X_train, X_val, y_train, y_val = split_data(df=df, test_size=hp.test_size, random_state=hp.random_state)
    lgb_train, lgb_eval = load_data(X_train, y_train, X_val, y_val)
    params = get_params(num_leaves=hp.num_leaves, learning_rate=hp.learnnig_rate)
    return train_model(params=params, lgb_train=lgb_train, lgb_eval=lgb_eval)


# # predict test data
# y_pred = model.predict(X_test)

# # accuracy check
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse**(0.5)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % rmse)

if __name__=="__main__":
    run_wf(hp=hp)






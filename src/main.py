# import all dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Tuple
from sidetrek.types.dataset import SidetrekDataset
from sidetrek.dataset import load_dataset

@dataclass_json
@dataclass 
class Hyperparameters(object):
    test_size: float = 0.15
    val_size: float = 0.10
    random_state: int = 786
    num_leaves: int = 10
    learning_rate: float = 0.05


hp = Hyperparameters()



# collect data
def get_data(ds: SidetrekDataset) -> pd.DataFrame:
    data = load_dataset(ds=ds, data_type="csv")
    data_dict = {}
    cols = list(data)[0]
    for k,v in enumerate(data):
        if k>0:
            data_dict[k]=v 
    
    df = pd.DataFrame.from_dict(data_dict, columns=cols, orient="index")
    type_dict = {
        "id" : int,
        "carat" : float,
        "cut" : str,
        "color" : str,
        "clarity" : str,
        "depth" : float,
        "table" : float,
        "x" : float,
        "y" : float,
        "z" : float,
        "price" : int,
    }
    df = df.astype(type_dict)
    return df

# preprocess data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["cut","color","clarity"]
    df.drop(["id"], axis=1, inplace=True)
    df = pd.get_dummies(data=df, columns=cat_cols)
    return df

# split dataset
def split_data(hp: Hyperparameters, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(["price"], axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=hp.test_size, random_state=hp.random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=hp.val_size, random_state=hp.random_state)
    return X_train, X_val, y_train, y_val 


# train model
def train_model(hp: Hyperparameters, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train:pd.Series, y_val: pd.Series) -> lgb.basic.Booster:
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {
        'task': 'train', 
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': hp.num_leaves,
        'learnnig_rate': hp.learning_rate,
        'metric': {'l2','l1'},
        'verbose': -1
    }
    return lgb.train(params, train_set=lgb_train, valid_sets=lgb_eval)


# # run workflow
# def run_wf(hp: Hyperparameters)->lgb:
#     df = get_data(filepath=hp.filepath)
#     df = preprocess_data(df=df, cat_cols=cat_cols)
#     X_train, X_val, y_train, y_val = split_data(df=df, test_size=hp.test_size, random_state=hp.random_state)
#     lgb_train, lgb_eval = load_data(X_train, y_train, X_val, y_val)
#     params = get_params(num_leaves=hp.num_leaves, learning_rate=hp.learnnig_rate)
#     return train_model(params=params, lgb_train=lgb_train, lgb_eval=lgb_eval)


# # predict test data
# y_pred = model.predict(X_test)

# # accuracy check
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse**(0.5)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % rmse)

# if __name__=="__main__":
#     run_wf(hp=hp)






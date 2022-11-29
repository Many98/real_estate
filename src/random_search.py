import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import max_error
from sklearn.metrics import median_absolute_error
from matplotlib.pyplot import figure
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from numpy import array

def data(dataset):
  df = pd.read_csv(dataset, sep=',', delimiter=None, encoding="utf8")
  # drop indexes
  df.drop(columns=df.columns[0], axis=1, inplace=True) 
  # drop non-usable columns
  df.drop("floor_area", axis=1, inplace=True) 
  df.drop("desc_hash", axis=1, inplace=True) 
  df.drop("hash", axis=1, inplace=True) 
  # drop texts columns
  df.drop("description", axis=1, inplace=True) # from nlp
  df.drop("note", axis=1, inplace=True) # from nlp
  df.drop("header", axis=1, inplace=True) # from nlp
  df.drop("heating_txt", axis=1, inplace=True)
  df.drop("waste_txt", axis=1, inplace=True)
  df.drop("telecomunication_txt", axis=1, inplace=True)
  df.drop("electricity_txt", axis=1, inplace=True)
  df.drop("tags", axis=1, inplace=True)
  df.drop("geometry", axis=1, inplace=True)
  df.drop("additional_disposition", axis=1, inplace=True) 
  df.drop("transport", axis=1, inplace=True) 
  df.drop("name", axis=1, inplace=True) 
  # df.drop("name", axis=1, inplace=True) 
  df.drop("place", axis=1, inplace=True) # praha 1 apod. isn't in data frame
  df.drop("date", axis=1, inplace=True) # later transform into float (2018+2/12)
  # for error feature_names may not contain [, ] or <
  df.columns = df.columns.str.replace(r'.', '_')
  df.columns = df.columns.str.replace(r'-', '_')
  df.columns = df.columns.str.replace(r'>=', '_vetsi_rovno_')
  df.columns = df.columns.str.replace(r'+', '_plus_')
  df.columns = df.columns.str.replace(r' ', '_')
  df.columns = df.columns.str.replace(r'/', '_')
  df.columns = df.columns.str.replace(r'[', '_')
  df.columns = df.columns.str.replace(r']', '_')
  df.columns = df.columns.str.replace(r'<', 'mensi')
  # log transformation
  df['price'] = np.log(df['price'])
  return df

def search(X_train, X_test, Y_train, Y_test):
  params = {
    'boosting_type':['gbdt', 'dart', 'goss'],
    'num_leaves':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_depth': [2,3,4,6,7],
    'learning_rate':[i/100 for i in range(1,10)],
    'n_estimators':[i*100 for i in range(1,10)],
    'min_child_samples':[5,10,15,20,25,30],
    'gamma':[i/10.0 for i in range(3,6)],  
    'subsample':[i/10.0 for i in range(1,10)],
    'colsample_bytree':[i/10.0 for i in range(6,11)], 
    'objective': ['reg:squarederror', 'reg:tweedie'],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eval_metric': ['rmse'],
    'eta': [i/10.0 for i in range(3,6)],
    'validate_parameters' : [True, False],
    'tree_method' : ['approx', 'hist', 'gpu_hist', 'approx', 'gpu_hist'],
  }
  reg = XGBRegressor(nthread=-1)
  n_iter_search = 200
  random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_search, cv=5, scoring='neg_mean_squared_error')
  random_search.fit(X_train, Y_train)
  best_regressor = random_search.best_estimator_
  final_model = best_regressor
  final_model.fit(X_train, Y_train)
  y_pred = final_model.predict(X_test)
  print("The model training score is " , final_model.score(X_train, Y_train))
  print("The model testing score is " , final_model.score(X_test, Y_test))
  print("The model testing mean absolute error is ", mean_absolute_error(np.exp(Y_test), np.exp(y_pred)))
  print("The model max error is ", max_error(np.exp(Y_test), np.exp(y_pred)))
  print("The model median absolute error is ", median_absolute_error(np.exp(Y_test), np.exp(y_pred)))

df = data('test_dataset.csv')

# columns with embeddings
only_embeddings = [i for i in df.columns if 'emb' in i]
df = df.drop([emb for emb in only_embeddings], axis=1)
# only ordinal columns 
only_ord = [i for i in df.columns if 'ord' in i]

# columns with ordinal distance
only_ord_dist = [i for i in df.columns if 'dist' in i and 'ord' in i and 'num' not in i and 'city' not in i]
# only numerical dist columns (with indicator >=1500m)
only_num_dist = [i for i in df.columns if 'dist' in i and 'ord' not in i and 'num' in i and 'city' not in i] + \
                [i for i in df.columns if '1500m' in i]
# only one-hot encoded distances
only_one_hot_dist = [i for i in df.columns if 'dist' in i and 'ord' not in i and
                     'num' not in i and 'city' not in i]
only_one_hot_100m = [i for i in df.columns if '0-99m' in i]
# only coordinates
only_coords = ['long', 'lat']
# Gaussian process predicition
only_gp = [i for i in df.columns if 'gp' in i]
# noise
only_noise = [i for i in df.columns if 'noise' in i]
# quality
only_quality_ord = [i for i in df.columns if ('quality' in i or
                'sun' in i or 'built' in i) and 'ord' in i]
# one hotncoded energy effeciency
only_energy = [i for i in df.columns if 'energy_effeciency' in i and 'ord' not in i]
# disposition
only_disposition = [i for i in df.columns if 'disposition' in i]
# construction_type
only_construction = [i for i in df.columns if 'construction_t' in i]
# ownership
only_ownership = [i for i in df.columns if 'ownership' in i]
# equipment
only_equipment = [i for i in df.columns if 'equipment' in i]
# state
only_state = [i for i in df.columns if 'state' in i]
# only has_<> features
only_has = [i for i in df.columns if 'has' in i]

df_final = df[['price', 'usable_area',  
               #'year_reconstruction_ord',
               'floor',] + only_gp + only_ord + only_num_dist + only_disposition + only_energy + only_ownership + only_equipment + only_state + only_has + only_construction].copy()
df_final = df_final.drop([ord for ord in only_ord_dist], axis=1)
Y_final = df_final['price'].values.reshape(-1,1)
X_final = df_final.drop("price", axis=1)
X_train_final, X_test_final, Y_train_final, Y_test_final = train_test_split(X_final, Y_final, test_size=0.2, random_state=42, shuffle=True)
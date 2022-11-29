import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def xgb_tune(X, y, njobs=5):
    params = {
        'max_leaves': [0, 10, 20, 50, 100],
        'max_depth': [2, 3, 4, 5, 6, 7, 10, 15],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.08999, 0.1, 0.11, 0.12, 0.15, 0.25, 0.3],
        'n_estimators': [i * 100 for i in range(1, 10)],
        'gamma': [0, 0.1, 1, 10, 100],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'colsample_bynode': [i / 10.0 for i in range(6, 11)],
        'booster': ['gbtree', 'dart', 'gblinear'],
        #'eval_metric': ['rmse'],
        #'tree_method': ['approx', 'hist'],
    }
    reg = XGBRegressor(nthread=None, njobs=None,
                       objective='reg:squarederror',
                       tree_method='hist',
                       random_state=42)
    n_iter_search = 5000
    random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_search, cv=5, n_jobs=njobs,
                                       scoring='r2', #'neg_mean_absolute_error',
                                       verbose=1)

    random_search.fit(X, y)

    return random_search.best_estimator_


def xgb_quantile_tune(df: pd.DataFrame):
    pass

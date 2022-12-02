import pandas as pd
from xgboost import XGBRegressor
from models.XGBQuantile import XGBQuantile
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import scipy


def xgb_tune(X, y, njobs=5):
    params = {
        'max_leaves': [0, 50, 100],
        'max_depth': [3, 4, 5, 6, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.08999, 0.1, 0.11, 0.12, 0.15, 0.25],
        'n_estimators': [i * 100 for i in range(1, 10)],
        'gamma': [0, 0.1, 1, 10, 100],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'colsample_bynode': [i / 10.0 for i in range(6, 11)],
        'booster': ['gbtree', 'dart'],
    }
    reg = XGBRegressor(nthread=None, njobs=None,
                       objective='reg:squarederror',
                       tree_method='hist',
                       random_state=42)
    n_iter_search = 1000
    random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_search, cv=5, n_jobs=njobs,
                                       scoring='neg_mean_absolute_error',
                                       verbose=1)

    random_search.fit(X, y)

    return random_search.best_estimator_


def xgb_quantile_tune(X, y, alpha, njobs=5):
    params = {
         'quant_delta': scipy.stats.uniform(0.01, 10.0),
         'quant_var': scipy.stats.uniform(1.0, 10.0),
         'quant_thres': scipy.stats.uniform(0.01, 10.0)

    }

    reg = XGBQuantile(
                    quant_alpha=alpha,
                    #quant_delta= 1.0,
                    #quant_thres = 1.0 ,quant_var=1.0,
                    n_estimators=800,
                    learning_rate=0.05,
                    colsample_bytree=1.0,
                    colsample_bynode=0.60,
                    # objective='reg:squarederror',
                    # eval_metric=mean_absolute_error,
                    max_depth=8,
                    tree_method='hist',
                    # enable_categorical=True,
                    subsample=0.8,
                    random_state=42,
                    # silent=True,
                    booster='dart',
                    n_jobs=njobs
                )
    n_iter_search = 100
    random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_search, cv=5, n_jobs=njobs,
                                        # 'neg_mean_absolute_error',
                                       verbose=1)

    random_search.fit(X, y)

    return random_search.best_estimator_

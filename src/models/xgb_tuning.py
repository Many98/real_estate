import pandas as pd
import numpy as np
import sklearn.compose
from xgboost import XGBRegressor
from models.XGBQuantile import XGBQuantile
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import scipy


def xgb_tune(pipe: Pipeline, X, y, n_iter_search=1000, njobs=5):
    params = {
        #'max_leaves': [0, 50, 100],
        'max_depth': [3, 4, 5
                      #  , 6, 7, 10, 15
                      ],
        'learning_rate': [0.08999, 0.1, 0.11, 0.12, 0.15
                          ],
        'n_estimators': [i * 100 for i in range(1, 10)],
        'gamma': [0, 0.1, 1],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'colsample_bynode': [i / 10.0 for i in range(6, 11)]
    }

    if isinstance(pipe['model'], sklearn.compose.TransformedTargetRegressor):
        params = {'model__regressor__' + k: v for k, v in params.items()}
    elif isinstance(pipe['model'], XGBRegressor):
        params = {'model__' + k: v for k, v in params.items()}
    else:
        raise Exception(f'Model should of type `sklearn.compose.TransformedTargetRegressor` or '
                        f'`xgboost.XGBRegressor`')

    random_search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=n_iter_search,
                                       cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
                                       #cv=10,
                                       n_jobs=njobs,
                                       scoring='neg_mean_absolute_error',
                                       verbose=4)

    random_search.fit(X, y)

    return random_search.best_estimator_


def xgb_quantile_tune(pipe: Pipeline, X, y, n_iter_search=100, njobs=5):
    params = {
         'quant_delta': scipy.stats.uniform(0.01, 10.0),
         'quant_var': scipy.stats.uniform(1.0, 10.0),
         'quant_thres': scipy.stats.uniform(0.01, 10.0)

    }
    if isinstance(pipe['model'], sklearn.compose.TransformedTargetRegressor):
        params = {'model__regressor__' + k: v for k, v in params.items()}
    elif isinstance(pipe['model'], XGBQuantile):
        params = {'model__' + k: v for k, v in params.items()}
    else:
        raise Exception(f'Model should of type `sklearn.compose.TransformedTargetRegressor` or '
                        f'`XGBQuantile`')

    random_search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=n_iter_search,
                                       cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42), n_jobs=njobs,
                                        # 'neg_mean_absolute_error',
                                       verbose=2)

    random_search.fit(X, y)

    return random_search.best_estimator_

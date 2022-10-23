import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import train_test_split, GridSearchCV
from src.preprocessing.preprocessing import prepare_atlas_cen_data
import numpy as np
from typing import Union
import os
import pickle

# TODO see https://towardsdatascience.com/tree-boosting-for-spatial-data-789145d6d97d
#  GPBoost kind of mix of gradient boosted trees with gaussian processes
#  because XGboost alone cannot accounts for autocorrelation of residuals in spatial data (but GP can)
#  maybe using output of gaussian process as input to XGboost can help XGB to handle spatial autocorrelation of prices


def gp_train(grid: Union[list, dict], bbox: tuple = (14.0, 14.8, 49.9, 50.3),
             csv_path: str = '../data/_atlas_cen_scraped.csv') -> tuple:
    """
    fit gaussian process on location data
    Parameters
    ----------
    grid: Union[list, dict]
        Defines state space for hyperparams to be tested
    bbox : tuple
        Bounding box (x_min, x_max, y_min, y_max)
    csv_path: str

    Returns
    -------

    """
    data = prepare_atlas_cen_data(csv_path)

    X, y = data[['long', 'lat']].to_numpy(), data['price/m2'].to_numpy()

    # partition data based on price and bbox
    valid_ids_low = np.where(((X[:, 0] > bbox[0]) & (X[:, 0] < bbox[1]) & (X[:, 1] < bbox[3]) & (X[:, 1] > bbox[2])
                              & (y < 250000)))[0]
    valid_ids_high = np.where(((X[:, 0] > bbox[0]) & (X[:, 0] < bbox[1]) & (X[:, 1] < bbox[3]) & (X[:, 1] > bbox[2])
                               & (y >= 250000)))[0]

    X_low, y_low = X[valid_ids_low], y[valid_ids_low]
    X_high, y_high = X[valid_ids_high], y[valid_ids_high]
    X_all, y_all = np.concatenate([X_low, X_high]), np.concatenate([y_low, y_high])

    # train/test split
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, random_state=0)

    # fit GP on all data points
    gpr_cv_all = GridSearchCV(estimator=GaussianProcessRegressor(), param_grid=grid, cv=5, n_jobs=4, verbose=2)
    gpr_cv_all.fit(X_train_all, y_train_all)

    # fit GP on "low" price data points
    gpr_cv_low = GridSearchCV(estimator=GaussianProcessRegressor(), param_grid=grid, cv=5, n_jobs=4, verbose=2)
    gpr_cv_low.fit(X_train_low, y_train_low)

    return gpr_cv_all, gpr_cv_low


def gp_inference(X: Union[np.ndarray, pd.DataFrame], model_path: str, data_path: str =
                 '../data/_atlas_cen_scraped.csv'):
    """
    Performs prediction using pickled model
    Parameters
    ----------
    X :
    model_path :
    data_path:
    Returns
    -------

    """
    if os.path.isfile(model_path):
        gp_model = pickle.load(open(model_path, 'rb'))
    else:
        raise Exception('model not found')

    data = prepare_atlas_cen_data(data_path)

    X, y = data[['long', 'lat']].to_numpy(), data['price/m2'].to_numpy()

    mean_pred, std_pred = gp_model.predict(X, return_std=True)

    ci_high_pred = mean_pred + 2 * std_pred
    ci_low_pred = mean_pred - 2 * std_pred

    # fix when variance is so high that ci_lower bound is under zero which is not wanted
    fix = np.where(ci_low_pred <= 0)[0]
    ci_low = np.where(fix, np.quantile(y, 0.1), ci_low_pred)
    mean = np.where(fix, (ci_high_pred + ci_low) / 2, mean_pred)

    return mean, std_pred, ci_low, ci_high_pred


if __name__ == '__main__':
    # define hyper-param space for brute force "optimization"
    # its done in few steps because of lack of computing resources
    # RBF performs poorly best cross val score is ~ 0.23
    """
    grid1 = [
        {
            "normalize_y": [True, False],
            "kernel": [RBF(length_scale=l, length_scale_bounds="fixed") for l in np.logspace(-5, 1, 7)]
        }
    ]
    model1 = gp_train(grid=grid1, csv_path='../../data/_atlas_cen_scraped.csv')
    """
    # Matern kernel is generalization of RBF
    grid2 = {
        "normalize_y": [True],
        "kernel": [Matern(length_scale=l, length_scale_bounds="fixed", nu=n) for l in
                   [0.1, 0.01, 0.001, 0.0001, 0.0008] for n in
                   [0.001, 0.01, 0.1, 0.2, 0.5]]
    }
    model2 = gp_train(grid=grid2, csv_path='../../data/_atlas_cen_scraped.csv')

    # best models was {'kernel': Matern(length_scale=0.1, nu=0.01), 'normalize_y': True} on "low" prices

    print('f')


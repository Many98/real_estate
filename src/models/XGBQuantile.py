from xgboost import XGBRegressor
import numpy as np
from functools import partial


def coverage_fraction(y, y_low, y_high):
    """
    auxiliary function to estimate coverage fraction of quantile regression
    i.e. whether it really estimates e.g. 90% confidence interval
    Parameters
    ----------
    y : np.ndarray
    y_low : np.ndarray
    y_high : np.ndarray

    Returns
    -------

    """
    return np.mean(np.logical_and(y >= y_low, y <= y_high))


class XGBQuantile(XGBRegressor):
    def __init__(self, quant_alpha=0.5, quant_delta=1.0, quant_thres=1.0,
                 quant_var=1.0, base_score=0.5,
                 booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, colsample_bynode=1, gamma=0,
                 learning_rate=0.1, max_delta_step=0, max_depth=3,
                 min_child_weight=1,
                 tree_method='hist',
                 missing=np.nan,
                 n_estimators=100,
                 n_jobs=0,
                 #nthread=0,
                 objective='reg:squarederror',
                 eval_metric=None, early_stopping_rounds=None,
                 random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 enable_categorical=False,
                 gpu_id=-1,
                 grow_policy='depthwise',
                 importance_type=None,
                 interaction_constraints='',
                 max_bin=256,
                 max_cat_to_onehot=4,
                 max_leaves=0,
                 monotone_constraints='()',
                 num_parallel_tree=1,
                 predictor='auto',
                 sampling_method='uniform',
                 validate_parameters=1,
                 verbosity=None,
                 feature_types=None,
                 max_cat_threshold=64,
                 #njobs=None,
                 callbacks=None,
                 subsample=1):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super().__init__(base_score=base_score, booster=booster,
                         colsample_bylevel=colsample_bylevel,
                         colsample_bytree=colsample_bytree,
                         colsample_bynode=colsample_bynode,
                         gamma=gamma, learning_rate=learning_rate,
                         max_delta_step=max_delta_step,
                         max_depth=max_depth, min_child_weight=min_child_weight,
                         tree_method=tree_method,
                         missing=missing,
                         n_estimators=n_estimators,
                         n_jobs=n_jobs,
                         #nthread=nthread,
                         objective=objective, random_state=random_state,
                         reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                         subsample=subsample,
                         early_stopping_rounds=early_stopping_rounds,
                         eval_metric=eval_metric,
                         enable_categorical=enable_categorical,
                         gpu_id=gpu_id,
                         callbacks=callbacks,
                         grow_policy=grow_policy,
                         importance_type=importance_type,
                         interaction_constraints=interaction_constraints,
                         max_bin=max_bin,
                         max_cat_to_onehot=max_cat_to_onehot,
                         max_leaves=max_leaves,
                         monotone_constraints=monotone_constraints,
                         num_parallel_tree=num_parallel_tree,
                         predictor=predictor,
                         feature_types=feature_types,
                         max_cat_threshold=max_cat_threshold,
                         sampling_method=sampling_method,
                         validate_parameters=validate_parameters,
                         verbosity=verbosity,
                         #njobs=njobs,
                         )

        self.test = None

    def fit(self, X, y
            # ,eval_set=None
            ):
        super().set_params(objective=partial(XGBQuantile.quantile_loss,
                                             alpha=self.quant_alpha,
                                             delta=self.quant_delta,
                                             threshold=self.quant_thres,
                                             var=self.quant_var
                                             ))
        super().fit(X, y
                    # , eval_set=eval_set
                    )
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1. / score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred

        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true, y_pred, alpha, delta):
        x = y_true - y_pred

        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - \
               ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - \
               alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (
                    np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))

        return np.array(split_gain)

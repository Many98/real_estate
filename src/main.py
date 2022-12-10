import pandas as pd
import numpy as np
import json
import os
import pickle
import joblib
import requests
import py7zr

from typing import Any

import xgboost
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, max_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from category_encoders import TargetEncoder, SummaryEncoder, QuantileEncoder

import shap

from preprocessing.enrichment import Enricher, Generator
from preprocessing.preprocessing import Preprocessor
from preprocessing.synchronization import Synchronizer
from scrapers.crawl_sreality import KindOfCrawlerForSReality
from scrapers.crawl_breality import KindOfCrawlerForBezRealitky
from scrapers.scrape_sreality import SRealityScraper
from scrapers.scrape_breality import BezRealitkyScraper
from scrapers.scrape_atlas_cen import AtlasCenScraper
from models.gaussian_process import gp_train
from models.xgb_tuning import xgb_tune
from xgboost import XGBRegressor
from models.XGBQuantile import XGBQuantile

import argparse
from typing import Union


class ETL(object):
    """class encapsulating whole preprocessing/ETL logic for data

        ?price map? | <crawl> -> [scrape] -> [synchronize] -> [enrich] -> [generate] -> [preprocess]  => pd.DataFrame
        Model generation will be independent of this ETL pipeline

    Overall steps in ETL should be:
    --------------------------
    0. ?price map? optionaly scrape Atlas cen to get prices of already sold apartments (and also fit gaussian process)
        this will be done asychronously with ETL

    1. <crawl>  crawl sreality/bezrealitky (regularly every week or so)
                if links are not provided as input via web app
    2. [scrape] Scrape all relevant (tabular/textual) data from links provided by crawlers/ provided by user as url
        Optionally process (web app) "manual" input i.e. user provides textual description and basic "tabular info"
                    like usable area, disposition, location etc.
    3. [synchronize] Synchronize attributes from sreality and bezrealitky data sources
    4. [enrich] Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc.
    5.  Feature engineering i.e.
       [generate] a) generation of additional/aggregate features
                        (requires research of e.g. econometrial methods)
       [preprocess] b) necessary preprocessing like handling missing values, one-hot encoding,
                       scaling features (if necessary e.g. for linear regression model) etc.
    ---------------------------------

    """

    def __init__(self, sreality_init_url: str = 'https://www.sreality.cz/hledani/prodej/byty/praha',
                 breality_init_url: str = 'https://www.bezrealitky.cz/vyhledat?offerType=PRODEJ&estateType=BYT&page=1&order=TIMEORDER_DESC&regionOsmIds=R435541&osm_value=Praha%2C+%C4%8Cesko',
                 crawled_links_filename: str = 'prodej_links.txt',
                 scrapped_data_filename: str = 'prodej',
                 inference: bool = False,
                 handmade: bool = False,
                 scrape: bool = False,
                 # load_dataset: bool = False,
                 ):

        self.inference = inference  # whether ETL is in INFERENCE phase
        self.handmade = handmade
        self.scrape = scrape  # whether to scrape data before processing
        # not functional in `inference` phase
        self.crawled_links_filename = crawled_links_filename
        self.scrapped_data_filename = scrapped_data_filename
        self.load_dataset = False

        if not self.inference and not self.scrape:
            if os.path.isfile('../data/dataset.csv'):
                self.load_dataset = True
            else:
                print('`scrape` parameter is set to False but not dataset found in \n'
                      '`../data/dataset.csv` therefore setting parameter scrape to True')
                self.scrape = True

        # TODO make inits more consistent
        self.breality_crawler = KindOfCrawlerForBezRealitky(out_filename=crawled_links_filename,
                                                            url=breality_init_url)
        self.sreality_crawler = KindOfCrawlerForSReality(out_filename=crawled_links_filename,
                                                         url=sreality_init_url)

        self.atlas_cen_scrapper = AtlasCenScraper()
        self.breality_scraper = BezRealitkyScraper()
        self.sreality_scraper = SRealityScraper()

        self.enricher = Enricher(df=pd.DataFrame())  # TODO not ideal init with empty dataframe

        # find out state of scrapped csv
        self.scraped_init_state = self._check_state()
        self.synchronizer = Synchronizer(from_row=self.scraped_init_state)

        self.generator = Generator(df=pd.DataFrame())  # TODO not ideal init with empty dataframe

        self.preprocessor = Preprocessor(df=pd.DataFrame(),
                                         inference=inference)  # TODO not ideal init with empty dataframe

    def __call__(self, update_price_map: bool = False, *args, **kwargs) -> dict:
        """
        call all instances of crawlers, scrapers, enricher etc.
        Parameters
        ----------
        update_price_map: bool
            whether to update price map
        Returns
        -------
        pd.DataFrame
        """
        # ### CLEANING OF TEMP CSV FILES
        self._clean()

        # ### 0 UPDATE PRICE MAP
        if update_price_map:
            self.update_price_map()

        # ### 1,2 OBTAIN RAW DATA
        if not self.inference:
            if not self.load_dataset:
                # links firstly obtained by crawlers and appended to `../data/prodej_links.txt`
                if self.scrape:
                    self.sreality_crawler.crawl()
                    self.breality_crawler.crawl()

                    # already scrapped links are appended to `../data/already_scraped_links.txt`
                    self.sreality_scraper.run(in_filename=self.crawled_links_filename,
                                              out_filename=self.scrapped_data_filename,
                                              inference=self.inference)
                    self.breality_scraper.run(in_filename=self.crawled_links_filename,
                                              out_filename=self.scrapped_data_filename,
                                              inference=self.inference)

                #  Data are now scrapped in two separate files
                #           `../data/prodej_breality.csv` and `../data/prodej_sreality.csv` so synchronization is needed
                #            to get one csv with same sets of attributes

                # ### 3 SYNCHRONIZE DATA
                data = self.synchronizer(
                    sreality_csv_path=f'../data/{self.scrapped_data_filename}_{self.sreality_scraper.name}_scraped.csv',
                    breality_csv_path=f'../data/{self.scrapped_data_filename}_{self.breality_scraper.name}_scraped.csv',
                    inference=self.inference)
            # Data are now synchronized in one ../data/tmp_synchronized.csv (TRAIN)
            else:
                data = pd.DataFrame()

        else:
            # input from user | used in inference

            if not self.handmade and os.path.isfile('../data/predict_links.txt'):
                self.sreality_scraper.run(in_filename='predict_links.txt', out_filename='predict',
                                          inference=self.inference)
                self.breality_scraper.run(in_filename='predict_links.txt', out_filename='predict',
                                          inference=self.inference)
                #  ### Data are now scrapped in two separate files
                #           ../data/predict_breality.csv and ../data/predict_sreality.csv so synchronization is needed
                #            to get one csv with same sets of attributes
            elif not self.handmade and not os.path.isfile('../data/predict_links.txt'):
                print('Links for prediction are not present')
                return {'data': None, 'quality_data': None, 'status': 'INTERNAL ERROR (OBTAIN)'}
                # raise Exception('Links for prediction are not present')

            # ### 3 SYNCHRONIZE DATA
            # TODO handle cases when empty df is returned
            # TODO for now handmade features are stored in `predict_breality_scraped.csv` it should be probably changed
            data = self.synchronizer(
                sreality_csv_path=f'../data/predict_{self.sreality_scraper.name}_scraped.csv',
                breality_csv_path=f'../data/predict_{self.breality_scraper.name}_scraped.csv', inference=self.inference)

            # Data are now synchronized in one ../data/tmp_synchronized.csv (INFERENCE)
        if self.inference or (not self.inference and not self.load_dataset):
            # TODO this expects self.inference be True only when used with web interface and it should be fixed
            if not self.load_dataset and data.empty and self.inference:
                print('Something went wrong. Data not obtained !')
                return {'data': None, 'quality_data': None, 'status': 'INTERNAL ERROR (SYNC)'}
            elif not self.load_dataset and data.empty and not self.inference:
                print('There are no new scraped data')

            # ### 4 ENRICH DATA
            self.enricher.df = data  # TODO not ideal
            enriched_data = self.enricher(inference=self.inference)

            if not self.load_dataset and enriched_data.empty and self.inference:
                print('Something went wrong. Data not obtained !')
                return {'data': None, 'quality_data': None, 'status': 'INTERNAL ERROR (ENRICH)'}

        if not self.inference and not self.load_dataset:
            self._export_data(enriched_data)
            self._set_state()
            dataset = pd.read_csv('../data/dataset.csv')
            # Data are now enriched with new geospatial attributes etc. and appended to`../data/dataset.csv`
        elif not self.inference and self.load_dataset:
            dataset = pd.read_csv('../data/dataset.csv')
        else:
            dataset = enriched_data

        # ### 5 a GENERATE AGGREGATED FEATURES (on-the-fly)
        # self.generator.df = dataset  # TODO not ideal
        # generated_data = self.generator()  #  embeddings are not used therefore generator is not needed

        self.preprocessor.df = dataset
        final_data = self.preprocessor()

        return final_data

    def update_price_map(self):
        """
        Update data from atlas cen and refit gaussian process
        Returns
        -------

        """
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
        """
        grid2 = {
            "normalize_y": [True],
            "kernel": [Matern(length_scale=l, length_scale_bounds="fixed", nu=n) for l in
                       [0.1, 0.01, 0.001, 0.0001, 0.0008] for n in
                       [0.001, 0.01, 0.1, 0.2, 0.5]]
        }
        model2 = gp_train(grid=grid2, csv_path='../../data/_atlas_cen_scraped.csv')
        """
        # best models was {'kernel': Matern(length_scale=0.1, nu=0.01), 'normalize_y': True} on "low" prices
        pass

    def _export_data(self, data: pd.DataFrame) -> None:
        """
        method to export final dataframe to csv database
        """
        if not data.empty:
            data.to_csv("../data/dataset.csv", mode='a', index=False,
                        header=not os.path.exists("../data/dataset.csv"))

        print(f'New data appended successfully in {"../data/dataset.csv"}!', end="\r", flush=True)

    def _set_state(self):
        s_state = 0
        b_state = 0
        try:
            sreality_scrapped = pd.read_csv(
                f'../data/{self.scrapped_data_filename}_{self.sreality_scraper.name}_scraped.csv', usecols=['header'])
            s_state = sreality_scrapped.shape[0]
        except:
            pass
        try:
            breality_scrapped = pd.read_csv(
                f'../data/{self.scrapped_data_filename}_{self.breality_scraper.name}_scraped.csv', usecols=['header'])
            b_state = breality_scrapped.shape[0]
        except:
            pass

        with open('preprocessing/sync_state.json', 'w') as f:
            json.dump({'sreality': s_state,
                       'breality': b_state}, f)

    def _check_state(self) -> tuple[int, int]:
        """
        auxiliary method to check state of scrapped csv files to find out index of last row
        ETL should also store own state dict in case that scraping was executed but other steps not
        Returns
        -------

        """
        if self.inference:
            return 0, 0
        else:
            if not os.path.isfile('preprocessing/sync_state.json'):
                self._set_state()
            with open('preprocessing/sync_state.json', 'r') as f:
                data = json.load(f)
            return data['sreality'], data['breality']

    def _clean(self):
        """
        auxiliary method to remove temp csv
        Returns
        -------

        """
        if os.path.isfile('../data/tmp_synchronized.csv'):
            os.remove('../data/tmp_synchronized.csv')
        if self.inference:
            if os.path.isfile('../data/predict_breality_scraped.csv') and not self.handmade:
                os.remove('../data/predict_breality_scraped.csv')
            if os.path.isfile('../data/predict_sreality_scraped.csv'):
                os.remove('../data/predict_sreality_scraped.csv')


class Model(object):
    """
    Encapsulates model fit/predict logic on prepared data
    [ETL] => pd.DataFrame -> [model] => prediction
    .
    .
    6. [model] Independent model generation handled by class `Model`

    """

    def __init__(self, data: pd.DataFrame, response: str = 'price_m2', log_transform: bool = False,
                 objective='reg:squarederror', n_iter_search=1000,
                 inference: bool = False, tune: bool = False):
        self.data = data

        # target encoding will replace nan with mean target value and that can be better than have artificial
        # level `unknown`
        self.data.replace('unknown', np.nan, inplace=True)

        self.response = response
        self.log_transform = log_transform
        self.inference = inference
        self.objective = objective
        self.n_iter_search = n_iter_search

        if not self.inference and self.response not in ['price', 'price_m2']:
            raise Exception(f"Response column must be one of ['price', 'price_m2']")

        if self.data.empty:
            raise Exception('Input dataset is empty')

        self.tune = tune

        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        self.explainer = None

    def __call__(self, *args, **kwargs) -> Union[tuple[Pipeline, Pipeline, Pipeline],
                                                 tuple[np.ndarray, np.ndarray, np.ndarray, Any]]:
        if not self.inference:
            # some theory https://www.kaggle.com/code/ryanholbrook/target-encoding/tutorial
            dist_cols = [i for i in self.data.columns if 'dist_te' in i]

            categorical_to_te = ['energy_effeciency', 'ownership', 'equipment', 'state', 'disposition',
                                 'construction_type', 'city_district'
                                 ]
            has_te = [i for i in self.data.columns if 'has_' in i and '_te' in i]

            categorical_to_tenc = has_te + categorical_to_te + dist_cols

            self.data[has_te] = self.data[has_te].astype('category')

            X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns.difference(['price',
                                                                                                        'price_m2',
                                                                                                        ])],
                                                                self.data[self.response], test_size=0.05,
                                                                random_state=42, shuffle=True)

            prep = ColumnTransformer([
                ('tenc', TargetEncoder(handle_unknown='value', handle_missing='value',
                                       min_samples_leaf=10, smoothing=5), categorical_to_tenc)
                # ('tenc', SummaryEncoder(quantiles=[0.25, 0.5, 0.75]), categorical_to_tenc)
                # ('tenc', QuantileEncoder(), categorical_to_tenc)
            ], remainder='passthrough')

            if self.tune:
                xgb = XGBRegressor(
                    objective=self.objective,
                    tree_method='hist',
                    booster='gbtree',
                    max_depth=4,
                    random_state=42)
            else:
                xgb = XGBRegressor(
                    n_estimators=700,
                    learning_rate=0.1,  # 0.08999,
                    colsample_bytree=0.7,  # 0.99,
                    colsample_bynode=0.9,  # 0.80,
                    subsample=0.8,  # 1
                    max_depth=4,  # 4
                    objective=self.objective,
                    tree_method='hist',
                    booster='gbtree',
                    grow_policy='depthwise',
                    # monotone_constraints={'energy_effeciency': 1, 'usable_area': 1},
                    random_state=42)

            # TODO XGBQuantile should be also tuned but for now this is enough
            xgbq_upper = XGBQuantile(quant_alpha=0.95,
                                     quant_delta=1.0,
                                     quant_thres=6.0,
                                     quant_var=3.2,
                                     n_estimators=100,
                                     learning_rate=0.08999,
                                     colsample_bytree=0.99,
                                     colsample_bynode=0.80,
                                     subsample=1,
                                     max_depth=4,
                                     tree_method='hist',
                                     booster='gbtree',
                                     random_state=42)

            xgbq_lower = XGBQuantile(quant_alpha=0.05,
                                     quant_delta=1.0,
                                     quant_thres=4.0,
                                     quant_var=4.2,
                                     n_estimators=100,
                                     learning_rate=0.08999,
                                     colsample_bytree=0.99,
                                     colsample_bynode=0.80,
                                     max_depth=4,
                                     tree_method='hist',
                                     booster='gbtree',
                                     random_state=42)

            tt_xgb = TransformedTargetRegressor(regressor=xgb, func=np.log, inverse_func=np.exp)
            tt_xgbq_lower = TransformedTargetRegressor(regressor=xgbq_lower, transformer=StandardScaler())
            tt_xgbq_upper = TransformedTargetRegressor(regressor=xgbq_upper, transformer=StandardScaler())

            self.model_mean = Pipeline([
                ('prep', prep),
                ('model', tt_xgb) if self.log_transform else ('model', xgb),
            ])

            self.model_upper = Pipeline([
                ('prep', prep),
                ('model', tt_xgbq_upper),  # XGBQuantile requires standardized response to work
            ])

            self.model_lower = Pipeline([
                ('prep', prep),
                ('model', tt_xgbq_lower),  # XGBQuantile requires standardized response to work
            ])

            if self.tune:
                self.model_mean = xgb_tune(self.model_mean, self.data[self.data.columns.difference(['price',
                                                                                                    'price_m2',
                                                                                                    ])],
                                           self.data[self.response],
                                           n_iter_search=self.n_iter_search)

            self.model_mean.fit(X_train, y_train)

            self.explainer = shap.Explainer(self.model_mean['model'], self.model_mean['prep'].transform(
                self.data[self.data.columns.difference(['price',
                                                        'price_m2'])]).astype(float))

            y_pred = self.model_mean.predict(X_test)

            print("The model training score is ", self.model_mean.score(X_train, y_train))
            print("The model testing score is ", self.model_mean.score(X_test, y_test))

            if self.response == 'price_m2':
                y_test2 = y_test * X_test['usable_area'].to_numpy()
                y_pred2 = y_pred * X_test['usable_area'].to_numpy()
            else:
                y_test2 = y_test
                y_pred2 = y_pred

            print("The model testing mean absolute error is ", mean_absolute_error(y_test2, y_pred2))
            print("The model max error is ", max_error(y_test2, y_pred2))
            print("The model median absolute error is ", median_absolute_error(y_test2, y_pred2))

            self.model_lower.fit(X_train, y_train)
            self.model_upper.fit(X_train, y_train)

            self.save_state()

        else:
            self.load_state()

            # TODO add more useful mapping to column/feature names
            out_cols = [i.replace('tenc__', '') for i in self.model_mean['prep'].get_feature_names()]

            shap_values = self.explainer(self.model_mean['prep'].transform(
                self.data[self.data.columns.difference(['price',
                                                        'price_m2'])]).astype(float)[:1])

            shap_values.feature_names = out_cols

            shap_values.data = shap_values.data.astype(object)
            shap_values.data[0] = self.data[out_cols].to_numpy()[0]
            #shap_values.values[0] *= self.data['usable_area'].to_numpy()[0]
            #shap_values.base_values[0] *= self.data['usable_area'].to_numpy()[0]

            y_pred_mean = self.model_mean.predict(self.data[self.data.columns.difference(['price',
                                                                                          'price_m2'])])
            y_pred_lower = self.model_lower.predict(self.data[self.data.columns.difference(['price',
                                                                                            'price_m2'])])
            y_pred_upper = self.model_upper.predict(self.data[self.data.columns.difference(['price',
                                                                                            'price_m2'])])

            if self.response == 'price_m2':
                y_pred_mean2 = y_pred_mean * self.data['usable_area'].to_numpy()
                y_pred_lower2 = y_pred_lower * self.data['usable_area'].to_numpy()
                y_pred_upper2 = y_pred_upper * self.data['usable_area'].to_numpy()
            else:
                y_pred_mean2 = y_pred_mean
                y_pred_lower2 = y_pred_lower
                y_pred_upper2 = y_pred_upper

            return y_pred_lower2, y_pred_mean2, y_pred_upper2, shap_values[0]
        # TODO we will need somehow deal with inconsistent predictions model_upper vs model_mean

    def load_state(self, path: str = None):
        # TODO repo is private therefore content cannot be downloaded -> make it public
        """
        if not os.path.isfile('models/xgb.pickle'):
            if not os.path.isfile('models/xgb.7z'):
                with requests.get('https://github.com/Many98/real_estate/raw/main/src/models/xgb.7z',
                                  stream=True) as r:
                    with open('models/xgb.7z', 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            with py7zr.SevenZipFile('models/xgb.7z', mode='r') as z:
                z.extractall(path='models/xgb.pickle')
        """
        if path is None:
            path = os.path.join('models', 'model.pkl')
        elif path is not None and not os.path.isfile(path):
            # TODO it should be downloaded first
            with py7zr.SevenZipFile(os.path.join('models', 'model.7z'), mode='r') as z:
                z.extractall(path='models/')
            path = os.path.join('models', 'model.pkl')

        # model = xgboost.XGBRegressor()
        # model.load_model('models/xgb.json')

        state = joblib.load(path)

        self.model_mean = state['model_mean']
        self.model_upper = state['model_upper']
        self.model_lower = state['model_lower']
        self.response = state['response']
        self.explainer = state['explainer']

    def save_state(self):
        state = {'model_mean': self.model_mean, 'model_upper': self.model_upper, 'model_lower': self.model_lower,
                 'response': self.response, 'explainer': self.explainer}

        joblib.dump(state, os.path.join('models', 'model.pkl'), compress=1)

        # self.model_mean['model'].regressor_.save_model('models/xgb.json')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Real estate scraper')
    # parser.add_argument('-c', '--config-name', help='Name of the config file', default='config.yaml')
    # arguments = parser.parse_args()

    parser = argparse.ArgumentParser(
        description="-------------------------------------- REAL ESTATE ------------------------------- \n"
                    "----------------------- Prediction of prices of apartments in Prague ---------------\n"

                    "Processing steps are: \n"
                    "1) Crawl & scrape data from sreality.cz and bezrealitky.cz or just load prepared dataset \n"
                    "2) Perform synchronization, enrichment and preprocessing\n"
                    "3) Run model in train/inference phase\n"
                    "  For better user experience use web interface \n"
                    "  Web interface can be run on localhost using `streamlit run web.py` command"
        ,

        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--train",
        action='store_true',
        help="Whether to run code in train phase (it will save resulting model in `models/model.pkl`). \n"
             "Note that not specifying this switch will result in loading dataset from \n"
             "`../data/dataset.csv` and performing prediction on it."
             "Result will be saved in `../data/result.csv`"

    )

    parser.add_argument(
        "--scrape",
        "-s",
        action='store_true',
        help="Whether to also scrape data otherwise prepared dataset is loaded. \n"
             "Note that scraping new data can take some time"

    )

    parser.add_argument(
        "--tune",
        action='store_true',
        help="Whether to run `train` phase and tune hyperparameters."

    )

    parser.add_argument(
        "--log_transform",
        "-log",
        action='store_true',
        help="Whether to apply log transform on response."

    )

    parser.add_argument(
        "--response",
        default='price_m2',
        help="Specifies type of response"
             "Can be one of [`price_m2`, `price`]"
    )

    parser.add_argument(
        "--objective",
        default='reg:squarederror',
        help="Specifies type of objective function used in XGB training. Default is L2 norm \n"
             "For details see https://xgboost.readthedocs.io/en/latest/parameter.html"
    )

    parser.add_argument(
        "--n_iter_search",
        default=100,
        help="Specifies number of samples from hyperparameter space to be evaluated during tuning. Default is 100 \n"
             "For details see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html"
    )

    args = parser.parse_args()


    # TODO in CLI I do not use args.train because it would raise error ... it should be fixed
    #  for now this will run ETL in train phase and load dataset and then Model in `not args.train` phase
    etl = ETL(scrape=args.scrape)

    out = etl()

    if out['status'] in ['EMPTY', 'RANP'] or 'INTERNAL ERROR' in out['status']:
        raise Exception(f'Data preprocessing failed with status `{out["status"]}`')
    else:
        print('DATA OBTAINED SUCCESSFULLY')

    model = Model(data=out['data'], inference=not args.train, tune=args.tune, response=args.response,
                  objective=args.objective,
                  n_iter_search=args.n_iter_search)
    # inference phase pd.DataFrame with features and predicted prices
    # train phase will returned path to serialized trained model
    trained = model()

    if not args.train:
        out['data']['lower_ci_xgb_prediction'] = trained[0]
        out['data']['mean_xgb_prediction'] = trained[1]
        out['data']['upper_ci_xgb_prediction'] = trained[2]

        out['data'][['lower_ci_xgb_prediction', 'mean_xgb_prediction', 'upper_ci_xgb_prediction',
                     'price', 'price_m2',
                     'gp_mean_price', 'usable_area']].to_csv('../data/result.csv')
        print('Prediction is save in `../data/result.csv`')

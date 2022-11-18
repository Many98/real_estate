import pandas as pd
import numpy as np
import json
import os
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

from preprocessing.enrichment import Enricher, Generator
from preprocessing.preprocessing import Preprocessor
from preprocessing.synchronization import Synchronizer
from scrapers.crawl_sreality import KindOfCrawlerForSReality
from scrapers.crawl_breality import KindOfCrawlerForBezRealitky
from scrapers.scrape_sreality import SRealityScraper
from scrapers.scrape_breality import BezRealitkyScraper
from scrapers.scrape_atlas_cen import AtlasCenScraper
from models.gaussian_process import gp_train

import argparse
from typing import Union


class ETL(object):
    """class encapsulating whole preprocessing/ETL logic for data

        ?price map? | <crawl> -> [scrape] -> [synchronize] -> [enrich] -> [generate] -> [preprocess]  => pd.DataFrame
        Model generation will be independent of this ETL pipeline

    Overall steps in ETL should be:
    --------------------------
    0. ?price map? optionaly scrape Atlas cen to get prices of already sold apartments (and also fit gaussian process)
        this will be done asychronously with ETL  --> DONE (Emanuel)

    1. <crawl>  crawl sreality/bezrealitky (regularly every week or so)  --> DONE (Emanuel)
                if links are not provided as input via web app  (partially DONE) --> TODO (Hanka)
    2. [scrape] Scrape all relevant (tabular/textual) data from links provided by crawlers/ provided by user as url
        Optionally process (web app) "manual" input i.e. user provides textual description and basic "tabular info"
                    like usable area, disposition, location etc.  --> DONE (Hanka & Adam)
    3. [synchronize] Synchronize attributes from sreality and bezrealitky data sources  --> DONE (Adam, Emanuel)
    4. [enrich] Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc.
            --> DONE (Emanuel)
    5.  Feature engineering i.e.
       [generate] a) generation of additional/aggregate features # TODO (Hanka)
                        (requires research of e.g. econometrial methods)  # TODO reserch (Hanka)
       [preprocess] b) necessary preprocessing like handling missing values, one-hot encoding,
                       scaling features (if necessary e.g. for linear regression model) etc. --> DONE (Emanuel)
    ---------------------------------

    """

    def __init__(self, sreality_init_url: str = 'https://www.sreality.cz/hledani/prodej/byty/praha',
                 breality_init_url: str = 'https://www.bezrealitky.cz/vyhledat?offerType=PRODEJ&estateType=BYT&page=1&order=TIMEORDER_DESC&regionOsmIds=R435541&osm_value=Praha%2C+%C4%8Cesko',
                 crawled_links_filename: str = 'prodej_links.txt',
                 scrapped_data_filename: str = 'prodej',
                 inference: bool = False):

        self.data = None
        self.inference = inference  # whether ETL is in INFERENCE phase
        self.crawled_links_filename = crawled_links_filename
        self.scrapped_data_filename = scrapped_data_filename

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

        self.preprocessor = Preprocessor(df=pd.DataFrame(), inference=inference)  # TODO not ideal init with empty dataframe

        self.generator = Generator(df=pd.DataFrame())  # TODO not ideal init with empty dataframe

    def __call__(self, update_price_map: bool = False, *args, **kwargs) -> pd.DataFrame:
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
            # links firstly obtained by crawlers and appended to `../data/prodej_links.txt`
            self.sreality_crawler.crawl()
            self.breality_crawler.crawl()

            # already scrapped links are appended to `../data/already_scraped_links.txt`
            self.sreality_scraper.run(in_filename=self.crawled_links_filename, out_filename=self.scrapped_data_filename,
                                      inference=self.inference)
            self.breality_scraper.run(in_filename=self.crawled_links_filename, out_filename=self.scrapped_data_filename,
                                      inference=self.inference)
            #  Data are now scrapped in two separate files
            #           `../data/prodej_breality.csv` and `../data/prodej_sreality.csv` so synchronization is needed
            #            to get one csv with same sets of attributes

            # ### 3 SYNCHRONIZE DATA
            # TODO handle cases when empty df is returned
            data = self.synchronizer(
                sreality_csv_path=f'../data/{self.scrapped_data_filename}_{self.sreality_scraper.name}_scraped.csv',
                breality_csv_path=f'../data/{self.scrapped_data_filename}_{self.breality_scraper.name}_scraped.csv')

            # Data are now synchronized in one ../data/tmp_synchronized.csv (TRAIN)

        else:
            # input from user | used in inference
            # TODO in future it shouldbe prepared to handle user input as text, own tabular data etc. but for now just
            #  links
            if os.path.isfile('../data/predict_links.txt'):
                self.sreality_scraper.run(in_filename='predict_links.txt', out_filename='predict',
                                          inference=self.inference)
                self.breality_scraper.run(in_filename='predict_links.txt', out_filename='predict',
                                          inference=self.inference)
                #  ### Data are now scrapped in two separate files
                #           ../data/predict_breality.csv and ../data/predict_sreality.csv so synchronization is needed
                #            to get one csv with same sets of attributes
            else:
                raise Exception('Links for prediction are not present')

            # ### 3 SYNCHRONIZE DATA
            # TODO handle cases when empty df is returned
            data = self.synchronizer(
                sreality_csv_path=f'../data/predict_{self.sreality_scraper.name}_scraped.csv',
                breality_csv_path=f'../data/predict_{self.breality_scraper.name}_scraped.csv')

            # Data are now synchronized in one ../data/tmp_synchronized.csv (INFERENCE)

        if data.empty:
            raise Exception('Something went wrong. Data not obtained !')

        # ### 4 ENRICH DATA
        self.enricher.df = data  # TODO not ideal
        enriched_data = self.enricher()

        if not self.inference:
            self._export_data(enriched_data)
            dataset = pd.read_csv('../data/dataset.csv')
            # Data are now enriched with new geospatial attributes etc. and appended to`../data/dataset.csv`
        else:
            dataset = enriched_data

        # ### 5 a GENERATE AGGREGATED FEATURES (on-the-fly)
        self.generator.df = dataset  # TODO not ideal
        generated_data = self.generator()

        # ### 5 b PREPROCESS DATA (on-the-fly)
        self.preprocessor.df = generated_data  # TODO not ideal
        preprocessed_data = self.preprocessor()

        if not self.inference:
            self._update_state()

        return preprocessed_data

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

    def _check_state(self) -> tuple[int, int]:
        """
        auxiliary method to check state of scrapped csv files to find out index of last row
        ETL should also store own state dict in case that scraping was executed but other steps not
        Returns
        -------

        """
        if not os.path.isfile('../data/etl_state.json') or self.inference:
            return 0, 0
        else:
            with open('../data/etl_state.json', 'r') as f:
                state = json.load(f)
            return state['state']

    def _update_state(self):
        """
        auxiliary method to update state of ETL object. state is meant as index of rows to continue
        Returns
        -------

        """
        # TODO handle cases when no scraped csv are present
        # TODO check if +-1 row is not needed
        try:
            sreality_scrapped = pd.read_csv(
                f'../data/{self.scrapped_data_filename}_{self.sreality_scraper.name}_scraped.csv')
            breality_scrapped = pd.read_csv(
                f'../data/{self.scrapped_data_filename}_{self.breality_scraper.name}_scraped.csv')
        except Exception('Something went wrong with loading of scraped.csv files. \n'
                         f'Files `../data/{self.scrapped_data_filename}_{self.sreality_scraper.name}_scraped.csv` and \n'
                         f'`../data/{self.scrapped_data_filename}_{self.breality_scraper.name}_scraped.csv` are'
                         f'probably not present') as e:
            print(e)
            raise

        state = {'state': (sreality_scrapped.shape[0], breality_scrapped.shape[0])}

        out = json.dumps(state, indent=4)

        with open("../data/etl_state.json", "w") as outfile:
            outfile.write(out)

    def _clean(self):
        """
        auxiliary method to remove temp csv
        Returns
        -------

        """
        if os.path.isfile('../data/tmp_synchronized.csv'):
            os.remove('../data/tmp_synchronized.csv')
        if self.inference:
            if os.path.isfile('../data/predict_breality_scraped.csv'):
                os.remove('../data/predict_breality_scraped.csv')
            if os.path.isfile('../data/predict_sreality_scraped.csv'):
                os.remove('../data/predict_sreality_scraped.csv')


class Model(object):
    """
    Encapsulates model fit/predict logic on prepared data
    [ETL] => pd.DataFrame -> [model] => prediction
    .
    .
    .
    6. [model] Independent model generation handled by class `Model` see below:
        fit final model/s
            * ideas to be tested are transform all textual data to tabular data (via embeddings to get representation) and fit probably XGboost
                also test if providing embeddings gives non-marginal boost of prediction skill
            * use more sophisticated methods to mine information from text
                -* use transformer to predict price then use ensemble on (xgboost+transformer)
                -* or use probably some transformer for keyword extraction to creaate tabular data of relevant data in text
                -* or just define some query words and measure some type of distances (edit distance, dot product)
                    between every word of text and query words (probably more robust than just regexing)

        TODO -- Hanka XGboost
             -- Adam Random forest
             -- Emanuel Electra (small-e-czech)
    """
    def __init__(self, data: pd.DataFrame, inference: bool = False):
        pass

    def __call__(self, *args, **kwargs) -> Union[str, pd.DataFrame]:
        pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Real estate scraper')
    # parser.add_argument('-c', '--config-name', help='Name of the config file', default='config.yaml')
    # arguments = parser.parse_args()

    etl = ETL(inference=False)
    final_data = etl()
    # TODO handle what to do when empty df
    # TODO handle correct state creation/updates
    # TODO prepare final data and perform final corrections and checks on `ETL` class
    # TODO unit-test / asserts sanity checks would be nice to have

    model = Model(data=final_data, inference=False)
    # inference phase pd.DataFrame with features and predicted prices
    # train phase will returned path to serialized trained model
    trained = model()


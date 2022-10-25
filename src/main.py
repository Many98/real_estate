from preprocessing.enrichment import Enricher
from scrapers.crawl_sreality import KindOfCrawlerForSReality
from scrapers.crawl_breality import KindOfCrawlerForBezRealitky
from scrapers.scrape_sreality import SRealityScraper
from scrapers.scrape_breality import BezRealitkyScraper
import argparse


class ETL(object):
    """class encapsulating whole preprocessing/ETL logic for data

        ?price map? | <crawl> -> [scrape] -> [enrich] -> [synchronize] -> [preprocess] -> [generate] => pd.DataFrame
        Model generation will be independent of this ETL pipeline

    Overall steps in ETL should be:
    --------------------------
    0. ?price map? optionaly scrape Atlas cen to get prices of already sold apartments (and also fit gaussian process)
        this will be done asychronously with ETL  --> DONE (Emanuel)

    1. <crawl>  crawl sreality/bezrealitky (regularly every week or so)  --> DONE (Emanuel)
                if links are not provided as input via web app  (partially DONE) --> TODO (Hanka but research of econometrial/real estate methods is priorite)
    2. [scrape] Scrape all relevant (tabular/textual) data from links provided by crawlers/ provided by user as url
        Optionally process (web app) "manual" input i.e. user provides textual description and basic "tabular info"
                    like usable area, disposition, location etc.  --> DONE (Hanka & Adam)
    3. [synchronize] Synchronize attributes from sreality and bezrealitky data sources  --> TODO (Adam)
    4. [enrich] Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc. # TODO (Emanuel)
    5.  Feature engineering i.e.
       [preprocess] a) necessary preprocessing like handling missing values, one-hot encoding,
                       scaling features (if necessary e.g. for linear regression model) etc. # TODO (Emanuel)
       [generate] b) generation of additional/aggregate features # TODO (Adama)
                        (requires research of e.g. econometrial methods)  # TODO reserch (Hanka)
    ---------------------------------

    """

    def __call__(self, *args, **kwargs):
        # TODO here call all instances of crawlers, scrapers, enricher etc.
        pass

    def update_price_map(self):
        """
        Update data from atlas cen and refit gaussian process
        Returns
        -------

        """
        pass


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

        TODO -- Hanka basic tabular model (LinearRegression, RandomForest, XGBoost)
             -- Emanuel & Adam independent textual models/embdeddings etc
    """


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Real estate scraper')
    # parser.add_argument('-c', '--config-name', help='Name of the config file', default='config.yaml')
    # arguments = parser.parse_args()

    # Here can be tested functionality of two scrapers
    b = BezRealitkyScraper()
    b.run(in_filename='prodej_links.txt', out_filename='prodej')
    s = SRealityScraper()
    s.run(in_filename='prodej_links.txt', out_filename='prodej')
    """
    sreality_crawler = KindOfCrawlerForSReality(out_filename='prodej_links.txt',
                                                url='https://www.sreality.cz/hledani/prodej/byty/praha?velikost=1%2Bkk,1%2B1&stav=pred-rekonstrukci,v-rekonstrukci')
    breality_crawler = KindOfCrawlerForBezRealitky(out_filename='prodej_links.txt',
                                                   url='https://www.bezrealitky.cz/vyhledat?offerType=PRODEJ&estateType=BYT&page=1&order=TIMEORDER_DESC&regionOsmId=R435514&osm_value=Hlavn%C3%AD+m%C4%9Bsto+Praha%2C+Praha%2C+%C4%8Cesko')
    sreality_crawler.crawl()
    breality_crawler.crawl()
    if sreality_crawler.reality_links:
        print(f'Found {len(sreality_crawler.reality_links)} new apartments at https://www.sreality.cz/!')
    if breality_crawler.reality_links:
        print(f'Found {len(breality_crawler.reality_links)} new apartments ar https://www.bezrealitky.cz/!')
    else:
        print('No new apartments found')
    """

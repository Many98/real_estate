from preprocessing.preprocessing import Enricher


class Pipeline(object):
    """class encapsulating whole preprocessing logic

    Steps in TRAIN phase should be:
    ----------------------
    0. scrape Atlas cen to get prices of already sold apartments (and also fit gaussian process)
    1. Crawl sreality/bezrealitky (regularly every week or so)
    2. Scrape all relevant (tabular/textual) data from links provided by crawlers
    3. Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc.
    4. Feature engineering i.e.
        a) necessary preprocessing like handling missing values, one-hot encoding, scaling features (if necessary e.g. for linear regression model) etc.
        b) generation of additional/aggregate features (requires research of e.g. econometrial methods)

    5. Model generation:
        fit final model/s
            * ideas to be tested are transform all textual data to tabular data (via embeddings to get representation) and fit probably XGboost
                also test if providing embeddings gives non-marginal boost of prediction skill
            * use more sophisticated methods to mine information from text
                -* use transformer to predict price then use ensemble on (xgboost+transformer)
                -* or use probably some transformer for keyword extraction to creaate tabular data of relevant data in text
                -* or just define some query words and measure some type of distances (edit distance, dot product)
                    between every word of text and query words (probably more robust than just regexing)

    Steps in INFERENCE phase should be:
    ----------------------
    1. Scrape all relevant (tabular/textual) data from links provided as input from app
       OR
       Process (web app) "manual" input i.e. user provides textual description and basic "tabular info" like usable area,
       disposition, location etc.

    2. Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc.
    3. Feature engineering i.e.
        a) necessary preprocessing like handling missing values, one-hot encoding, scaling features (if necessary e.g. for linear regression model) etc.
        b) generation of additional/aggregate features

    4. Use serialized models for prediction of price

    """
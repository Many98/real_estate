from preprocessing.preprocessing import Enricher


class Pipeline(object):
    """class encapsulating whole preprocessing logic

    Steps shall be:
    1. Crawl sreality/bezrealitky
    2.  a) Scrape all relevant (tabular/textual) data from links provided by crawlers / provided as input to final app
        b) scrape Atlas cen to get prices of already sold apartments
    3. Enrich records with additional features like noise levels, distance to nearest parks,
            level of criminality nearby, estimated price from gaussian process, embeddings for textual data etc.
    4. Feature engineering i.e.
        a) necessary preprocessing like handling missing values, one-hot encoding, scaling features (if necessary e.g. for linear regression model) etc.
        b) generation of additional/aggregate features (requires research of e.g. econometrial methods)

    5. Model generation:
        a) fit gaussian process on atlas cen data to get mean/variance price estimates
        b) fit final model/s
            * ideas to be tested are transform all textual data to tabular data (via embeddings to get representation) and fit probably XGboost
                also test if providing embeddings gives non-marginal boost of prediction skill
            * use more sophisticated methods to mine information from text
                -* use transformer to predict price then use ensemble on (xgboost+transformer)
                -* or use probably some transformer for keyword extraction to creaate tabular data of relevant data in text
                -* or just define some query words and measure some type of distances (edit distance, dot product) 
                    between every word of text and query words (probably more robust than just regexing)
    6. Predict...


    """
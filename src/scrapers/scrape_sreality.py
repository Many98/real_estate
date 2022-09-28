import pandas as pd
import csv
import os
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from scrapers.BaseScraper import BaseScraper


class SRealityScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'sreality') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> dict:
        """Here comes custom implementation for sreality
            E.g.  method must be able to scrape all relevant data from urls like
                https://sreality.cz/detail/prodej/byt/1+kk/praha-holesovice-veletrzni/3827836492
            Output should be dictionary e.g. {'header': 'Prodej bytu 1+1 43 m²',
                                              'price': 7 900 000,
                                              'text': '...',
                                              'stav_objektu': 'Před rekonstrukcí',
                                              'long': 45.1,
                                              'lat': 15.5} ... etc.
                dictionary is required 'cause it is straighforward to create pd.DataFrame on it and easily export to csv
                for details see `scrapers.BaseScraper`

            *    Here should be scraped all tabular data and also longitude and latitude  ...
                and returned as dictionary

            *    method should also retrieve urls of image data a then call `self._save_image(img_url, url)` (probably in for loop)
                ( save_image method will create unique hash for web_url which will serve as directory name for all images for
                that web_url)
            *  similarly for text data call self._save_text(text, url)

            * finally update dictionary  e.g. result.update({'hash': base64(url)}) (it will append particular hash (filename))
             to dictionary as we need reference where are stored images and text for specific url
                for particular `url`
        """
        if 'sreality' not in url:  # ensures correct link
            return {}
        else:
            self._save_image(
                             'https://d18-a.sdn.cz/d_18/c_img_gU_o/YBJSIh.jpeg?fl=res,749,562,3|wrm,/watermark/sreality.png,10|shr,,20|jpg,90',
                             url)
            self._save_text('blablablasfsdof', url)
            test_dict = {'header': 'Prodej bytu 4+kk • 123 m² bez realitky',
                         'price': 17850000,
                         'plocha': 123,
                         'stav_objektu': 'Před rekonstrukcí',
                         'long': 45.1,
                         'lat': 15.5,
                         'hash': 'asdafqwf6sadasew6'}
            return test_dict
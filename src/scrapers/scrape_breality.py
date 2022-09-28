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


class BezRealitkyScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'breality') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> dict:
        """Here comes custom implementation for bezrealitky
            E.g.  method must be able to scrape all relevant data from urls like
                https://www.bezrealitky.cz/nemovitosti-byty-domy/742622-nabidka-prodej-bytu
                Output should be dictionary e.g. {'header': 'Prodej bytu 4+kk • 123 m² bez realitky',
                                              'price': 17 850 000,
                                              'plocha': 123,
                                              'long': 45.1,
                                              'lat': 15.5} ... etc.

                dictionary is required 'cause it is straighforward to create pd.DataFrame on it and easily export to csv
                for details see `scrapers.BaseScraper`


            *    Here should be scraped all tabular data and also longitude and latitude (in bezrealitky its ussually
                as easy as looking for `lng` keyword in html code) ...
                and returned as dictionary


            *    method should also retrieve urls of image data a then call `self._save_image(img_url, url)` (probably in for loop)
                ( save_image method will create unique hash for web_url which will serve as directory name for all images for
                that web_url)
            *  similarly for text data call self._save_text(text, url)

            * finally update dictionary  e.g. result.update({'hash': self._hash(url)}) (it will append particular hash (filename))
             to dictionary as we need reference where are stored images and text for specific url
                for particular `url`

        """
        if 'bezrealitky' not in url:  # ensures correct link
            return {}
        else:
            self._save_image('https://api.bezrealitky.cz/media/cache/record_main/data/images/advert/730k/730486/1660896816-momyfdwydw-phpmalr4d.jpg', url)
            self._save_text('blablablasfsdof',url)
            test_dict = {'header': 'Prodej bytu 4+kk • 123 m² bez realitky',
                                                  'price': 17850000,
                                                  'plocha': 123,
                                                  'long': 45.1,
                                                  'lat': 15.5,
                                                   'hash': 'asdafqwf6ew6'}
            return test_dict


import pandas as pd
import csv
import os
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from bs4 import BeautifulSoup
import json
import re
import json
import urllib3
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
            driver.get(url)
            content = driver.page_source
            soup = BeautifulSoup(content)

            # JSON file scraping
            try:
                script = soup.find('script', attrs={'type': 'application/json'})
                string = script.text
                dict_ = json.loads(string)
                if 'origAdvert' not in list(
                        dict_['props']['pageProps'].keys()):  # advert was really deleted / status code 404
                    return {'status': 'expired'}
                dict_ = dict_['props']['pageProps']['origAdvert']
            except:
                print('Json loading failed')
            try:
                kk = dict_.get('poiData', None)
                dict_2 = json.loads(kk) if kk is not None else None
                dict_3 = dict_2.get('gps', None)

                out = {
                    'header': dict_.get('imageAltText', None),
                    # text description of disposition e.g. 3 + kk
                    'price': dict_.get('price', None),  # Celková cena
                    'note': None,
                    # poznamka (k cene) sometimes can have valid info like
                    # Při rychlém jednání možná sleva., včetně provize, včetně právního servisu, cena k jednání
                    'usable_area': dict_.get('surface', None),  # Užitná plocha
                    'floor_area': None,  # Plocha podlahová
                    'floor': dict_.get('etage', None),  # podlazie
                    'energy_effeciency': dict_.get('penb', None),
                    # Energetická náročnost (letters A-G) A=best, G=shitty
                    'ownership': dict_.get('ownership', None),
                    # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
                    'description': dict_.get('description', None),
                    'long': dict_3.get('lng', None),
                    'lat': dict_3.get('lat', None),
                    'hash': None,

                    # other - done
                    'gas': None,  # Plyn
                    'waste': None,  # Odpad
                    'equipment': dict_.get('equipped', None),  # Vybavení
                    'state': dict_.get('reconstruction', None),
                    # stav objektu e.g. po rekonstrukci/projekt etc  (10 states possible) see https://www.sreality.cz/hledani/byty
                    'construction_type': dict_.get('construction', None),
                    # Stavba (3 states possible ) panel, cihla, ostatni
                    'place': dict_.get('address', None),  # Umístění objektu
                    'electricity': None,  # elektrina
                    'heating': dict_.get('heating', None),  # topeni
                    'transport': None,  # doprava
                    'year_reconstruction': None,  # rok rekonstrukce
                    'telecomunication': None,  # telekomunikace

                    # binary info - done
                    'has_lift': dict_.get('lift', None),  # Výtah: True, False
                    'has_garage': dict_.get('garage', None),  # garaz
                    'has_cellar': dict_.get('cellar', None),  # sklep presence
                    'no_barriers': None,  # ci je bezbarierovy bezbarierovy
                    'has_loggia': dict_.get('loggia', None),  # lodzie
                    'has_balcony': dict_.get('balcony', None),  # balkon
                    'has_garden': dict_.get('frontGarden', None),  # zahrada
                    'has_parking': dict_.get('parking', None),

                    # additional info - sometimes
                    'cellar_area': dict_.get('cellarSurface', None),
                    # plocha sklepu (if provided)
                    'loggia_area': dict_.get('loggiaSurface', None),
                    'balcony_area': dict_.get('balconySurface', None),


                    # what has b reality in addition
                    'tags': '_'.join(dict_.get('tags', [])),
                    'disposition': dict_.get('disposition', None),
                    'age': dict_.get('age', None),
                    'condition': dict_.get('condition', None),
                    'is_new': dict_.get('newBuilding', None),

                    # binary civic amenities (obcanska vybavenost binarne info) - done
                    'MHD': None,
                    'train_station': None,
                    'post_office': None,
                    'atm': None,  # bankomat according to google translate :D
                    'bank': None,
                    'doctor': None,
                    'vet': None,
                    'school': None,
                    'kindergarten': None,
                    'supermarket_grocery': None,
                    'restaurant_pub': None,
                    'playground': None,
                    'sports_field': None,
                    # or similar kind of leisure amenity probably OSM would be better
                    'subway': None,
                    'tram': None,
                    'bus': None,
                    # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
                    'theatre_cinema': None,
                    'pharmacy': None,

                    # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) -
                    'bus_station_dist': None,
                    'train_station_dist': None,
                    'subway_station_dist': None,
                    'tram_station_dist': None,
                    'MHD_dist': None,
                    'post_office_dist': None,
                    'atm_dist': None,
                    'bank_dist': None,
                    'doctor_dist': None,
                    'vet_dist': None,
                    'primary_school_dist': None,
                    'kindergarten_dist': None,
                    'supermarket_grocery_dist': None,
                    'restaurant_pub_dist': None,
                    'playground_dist': None,
                    'sports_field_dist': None,
                    # or similar kind of leisure amenity probably OSM would be better
                    # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
                    'theatre_cinema_dist': None,
                    'pharmacy_dist': None

                }

                dj = dict_.get('dataJson', None)
                estim_price = json.loads(dj) if dj is not None else None

                if estim_price is not None:
                    out.update({'estimated_sale_price': estim_price.get('estimationSale', {}).get('price', None)})
                    out.update({'estimated_rent_price': estim_price.get('estimationRent', {}).get('price', None)})
                geo_data = {}
                if dict_2 is not None and dict_2 != []:
                    # TODO cannot be hardcoded as 1
                    geo_data = {
                        # binary civic amenities (obcanska vybavenost binarne info) - done
                        'MHD': True if 'public_transport' in list(dict_2.keys()) else None,  # spíš MHD
                        'post_office': True if 'post' in list(dict_2.keys()) else None,
                        'bank': True if 'bank' in list(dict_2.keys()) else None,
                        'school': True if 'school' in list(dict_2.keys()) else None,
                        'kindergarten': True if 'kindergarten' in list(dict_2.keys()) else None,
                        'supermarket_grocery': True if 'shop' in list(dict_2.keys()) else None,
                        'restaurant_pub': True if 'restaurant' in list(dict_2.keys()) else None,
                        'playground': True if 'playground' in list(dict_2.keys()) else None,
                        'sports_field': True if 'sports_field' in list(dict_2.keys()) else None,
                        'pharmacy': True if 'pharmacy' in list(dict_2.keys()) else None,

                        # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) -
                        'MHD_dist': dict_2.get('public_transport', {}).get('properties', {}).get('walkDistance', None),
                        # using `get` method it is more robust see https://www.w3schools.com/python/ref_dictionary_get.asp
                        'post_office_dist': dict_2.get('post', {}).get('properties', {}).get('walkDistance', None),
                        'bank_dist': dict_2.get('bank', {}).get('properties', {}).get('walkDistance', None),

                        'primary_school_dist': dict_2.get('school', {}).get('properties', {}).get('walkDistance', None),
                        'kindergarten_dist': dict_2.get('kindergartne', {}).get('properties', {}).get('walkDistance', None),
                        'supermarket_grocery_dist': dict_2.get('shop', {}).get('properties', {}).get('walkDistance', None),
                        'restaurant_pub_dist': dict_2.get('restaurant', {}).get('properties', {}).get('walkDistance', None),
                        'playground_dist': dict_2.get('playground', {}).get('properties', {}).get('walkDistance', None),
                        'sports_field_dist': dict_2.get('sports_field', {}).get('properties', {}).get('walkDistance', None),
                        'pharmacy_dist': dict_2.get('pharmacy', {}).get('properties', {}).get('walkDistance', None)

                    }

                out.update(geo_data)
                return out
            except KeyError:
                pass

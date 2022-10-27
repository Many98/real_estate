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
<<<<<<< HEAD
            # JSON file scraping
            script = soup.find('script', attrs={'type': 'application/json'})
            string = script.text
            dict = json.loads(string)
            # print(type(res)) is dict
            dict_2 = None
            try:
                dict_2 = json.loads(dict['props']['pageProps']['origAdvert']['poiData'])
                '''
                bus = False
                bus_dist = 0
                tram = False
                tram_dist = 0
                metro = False
                metro_dist = 0
                if 'tram' in dict_2['public_transport']['properties']['category_ids']['605']['category_name']:
                    tram = True
                    tram_dist = dict_2['public_transport']['properties']['walkDistance']
                if 'bus' in dict_2['public_transport']['properties']['category_ids']['605']['category_name']:
                    bus = True
                    bus_dist = dict_2['public_transport']['properties']['walkDistance']
=======

            script = soup.find('script', text=re.compile('{"props":{"pageProps"'))
            json_feed = str(script).split('<script id="__NEXT_DATA__" type="application/json">')[1].split('</script>')[0]
            data = json.loads(json_feed) # Contains all relevant data we need in very convenient json format ALSO FOR ALREADY NOT EXISTENT ADVERTS
            # TODO process json `data` extract all relevant info including osm data about nearby features and travel distance
            #   and map extracted attributes to ours `out` dict


            # scrape header
            header = soup.find('h1', attrs={'class': 'mb-3 mb-lg-10 h2'})[1].split('</script>')[0]
            header = str(header.text)
            usable_area = header.split(" ")[4] #in m^2
            header = header.split(" ")[2]
>>>>>>> 7222de27ea06a7c61412f3473ac8a2a3c17815ba

                if 'metro' in dict_2['public_transport']['properties']['category_ids']['605']['category_name']:
                    metro = True  # todo, if is it metro or subway!
                    metro_dist = dict_2['public_transport']['properties']['walkDistance'] '''

                # print(dict['props']['pageProps']['origAdvert']['poiData'])
                out = {'header': dict['props']['pageProps']['origAdvert']['imageAltText'],
                       # text description of disposition e.g. 3 + kk
                       'price': dict['props']['pageProps']['origAdvert']['price'],  # Celková cena
                       'note': dict['props']['pageProps']['origAdvert']['description'],
                       # poznamka (k cene) sometimes can have valid info like
                       # Při rychlém jednání možná sleva., včetně provize, včetně právního servisu, cena k jednání
                       'usable_area': dict['props']['pageProps']['origAdvert']['surface'],  # Užitná plocha
                       'floor_area': None,  # Plocha podlahová (?)
                       'floor': dict['props']['pageProps']['origAdvert']['etage'],  # podlazie
                       'energy_effeciency': dict['props']['pageProps']['origAdvert']['penb'],
                       # Energetická náročnost (letters A-G) A=best, G=shitty
                       'ownership': dict['props']['pageProps']['origAdvert']['ownership'],
                       # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
                       'description': dict['props']['pageProps']['origAdvert']['description'],
                       'long': dict['props']['pageProps']['origAdvert']['gps']['lng'],
                       'lat': dict['props']['pageProps']['origAdvert']['gps']['lat'],
                       'hash': None,

                       # binary civic amenities (obcanska vybavenost binarne info) - done
                       'MHD': 1,  # spíš MHD
                       'train_station': None,
                       'post_office': 1,
                       'atm': 1,  # bankomat according to google translate :D
                       'doctor': None,
                       'vet': None,
                       'primary_school': 1,
                       'kindergarten': 1,
                       'supermarket_grocery': 1,
                       'restaurant_pub': 1,
                       'playground': 1,
                       'sports_field': 1,
                       # or similar kind of leisure amenity probably OSM would be better
                       'subway': None,
                       'tram': None,
                       'bus': None,
                       # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
                       'theatre_cinema': None,
                       'pharmacy': 1,

                       # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) - TODO (?)
                       'bus_station_dist': None,
                       'train_station_dist': None,
                       'subway_station_dist': None,
                       'tram_station_dist': None,
                       'MHD_dist': dict_2['public_transport']['properties']['walkDistance'],
                       'post_office_dist': dict_2['post']['properties']['walkDistance'],
                       'atm_dist': dict_2['bank']['properties']['walkDistance'],
                       # bankomat according to google translate :D
                       'doctor_dist': None,
                       'vet_dist': None,
                       'primary_school_dist': dict_2['school']['properties']['walkDistance'],
                       'kindergarten_dist': dict_2['kindergarten']['properties']['walkDistance'],
                       'supermarket_grocery_dist': dict_2['shop']['properties']['walkDistance'],
                       'restaurant_pub_dist': dict_2['restaurant']['properties']['walkDistance'],
                       'playground_dist': dict_2['playground']['properties']['walkDistance'],
                       'sports_field_dist': dict_2['sports_field']['properties']['walkDistance'],
                       # or similar kind of leisure amenity probably OSM would be better
                       # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
                       'theatre_cinema_dist': None,
                       'pharmacy_dist': dict_2['pharmacy']['properties']['walkDistance'],

                       # other - done
                       'gas': None,  # Plyn
                       'waste': None,  # Odpad
                       'equipment': dict['props']['pageProps']['origAdvert']['equipped'],  # Vybavení
                       'state': dict['props']['pageProps']['origAdvert']['reconstruction'],
                       # stav objektu e.g. po rekonstrukci/projekt etc  (10 states possible) see https://www.sreality.cz/hledani/byty
                       'construction_type': dict['props']['pageProps']['origAdvert']['construction'],
                       # Stavba (3 states possible ) panel, cihla, ostatni
                       'place': dict['props']['pageProps']['origAdvert']['address'],  # Umístění objektu
                       'electricity': None,  # elektrina
                       'heating': dict['props']['pageProps']['origAdvert']['heating'],  # topeni
                       'transport': None,  # doprava
                       'year_reconstruction': None,  # rok rekonstrukce
                       'telecomunication': None,  # telekomunikace

                       # binary info - done
                       'has_lift': dict['props']['pageProps']['origAdvert']['balcony'],  # Výtah: True, False
                       'has_garage': dict['props']['pageProps']['origAdvert']['garage'],  # garaz
                       'has_cellar': dict['props']['pageProps']['origAdvert']['cellar'],  # sklep presence or  m2 ???
                       'no_barriers': None,  # ci je bezbarierovy bezbarierovy
                       'has_loggia': dict['props']['pageProps']['origAdvert']['loggia'],  # lodzie m2
                       'has_balcony': dict['props']['pageProps']['origAdvert']['balcony'],  # balkon
                       'has_garden': dict['props']['pageProps']['origAdvert']['frontGarden'],  # zahrada,
                       'has_parking': dict['props']['pageProps']['origAdvert']['parking'],

                       # additional info - sometimes
                       'cellar_area': dict['props']['pageProps']['origAdvert']['cellarSurface'],
                       # plocha sklepu (if provided)
                       'loggia_area': dict['props']['pageProps']['origAdvert']['loggiaSurface'],
                       'balcony_area': dict['props']['pageProps']['origAdvert']['balconySurface']}

                with open("data_brealitky.json", "a") as outfile:
                    json.dump(out, outfile)
                return out
            except KeyError:
                pass
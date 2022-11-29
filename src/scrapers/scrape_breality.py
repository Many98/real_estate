import time

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
import json
from scrapers.BaseScraper import BaseScraper


class BezRealitkyScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'breality') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> dict:
        """
        Here comes custom implementation for bezrealitky.cz
            E.g.  method must be able to scrape all relevant data from urls like
                https://www.bezrealitky.cz/nemovitosti-byty-domy/742622-nabidka-prodej-bytu
        Parameters
        ----------
        driver : WebDriver
            instance of selenium's WebDriver
        url : str
            url link to be scraped
        Returns
        -------
        dict with specific keys and scraped values
        """
        if 'bezrealitky' not in url:  # ensures correct link
            return {}
        else:
            driver.get(url)
            time.sleep(1.5)
            content = driver.page_source
            soup = BeautifulSoup(content, features="lxml")

            # JSON file scraping
            try:
                time.sleep(1)
                script = soup.find('script', attrs={'type': 'application/json'})
                string = script.text
                dict_ = json.loads(string)
                if 'origAdvert' not in list(
                        dict_['props']['pageProps'].keys()):  # advert was really deleted / status code 404
                    return {'status': 'expired'}
                dict_ = dict_['props']['pageProps']['origAdvert']
            except:
                print('Json loading failed')
                return {}
            try:
                kk = dict_.get('poiData', None)
                dict_2 = json.loads(kk) if kk is not None else None
                dict_3 = None
                if isinstance(dict_2, dict):
                    dict_3 = dict_2.get('gps', None)
                if dict_3 is None:
                    dict_3 = dict_.get('gps', {})

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
                    'state': dict_.get('condition', None),
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
                    'has_cellar': dict_.get('cellar', False) or dict_.get('cellarSurface', False),  # sklep presence
                    'no_barriers': None,  # ci je bezbarierovy bezbarierovy
                    'has_loggia': dict_.get('loggia', False) or dict_.get('loggiaSurface', False),  # lodzie
                    'has_balcony': dict_.get('balcony', False) or dict_.get('balconySurface', False),  # balkon
                    'has_garden': dict_.get('frontGarden', None),  # zahrada
                    'has_parking': dict_.get('parking', None),

                    # what has b reality in addition
                    'tags': '_'.join(dict_.get('tags', [])),
                    'disposition': dict_.get('disposition', None),

                    # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) -
                    'bus_station_dist': None,
                    'train_station_dist': None,
                    'subway_station_dist': None,
                    'tram_station_dist': None,
                    'post_office_dist': None,
                    'atm_dist': None,
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
                    'pharmacy_dist': None,
                    'name': self.name,
                    'date': datetime.today().strftime('%Y-%m-%d')

                }

                geo_data = {}
                if dict_2 is not None and dict_2 != []:
                    geo_data = {

                        # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) -
                        # using `get` method it is more robust see https://www.w3schools.com/python/ref_dictionary_get.asp
                        'post_office_dist': dict_2.get('post', {}).get('properties', {}).get('walkDistance', None),
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
                return {}

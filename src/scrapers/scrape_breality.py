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

            # price
            price = soup.find('strong', attrs={'class': 'h4 fw-bold'})
            price = str(price.text)
            price = price[:-3] #in czech crowns
            price = price.replace('\xa0', ' ')
            # where
            place = soup.find('a', attrs={'href': '#mapa'})
            place = str(place.text)

            # note
            description = soup.find('div', attrs={'class': 'box mb-8'})
            description = str(description.text)

            note = None
            if 'Sleva' in description:
                note = soup.find('p', attrs={'class': 'text-perex-lg'})


            # TODO
            # tabular_data - all
            table_data = []
            data = soup.find('div', attrs={'class': 'mb-last-0 vstack minBreakpoint-xs'})
            data = str(data.text)
            print(data)
            ownership = None
            if 'Osobní' in data:
                ownership = 1
            elif 'Družstevní' in data:
                ownership = 2
            elif 'Obecní' in data:
                ownership = 3
            else:
                ownership = None

            construction_type = None
            if 'Dřevostavba' in data:
                construction_type = 1
            elif 'Cihla' in data:
                construction_type = 2
            elif 'Panel' in data:
                construction_type = 3
            elif 'Nízkoenergetický' in data:
                construction_type = 4
            else:
                construction_type = None

            equipment = None
            if 'Vybaveno' in data:
                equipment = 1
            elif 'Částečně' in data:
                equipment = 2
            elif 'Nevybaveno' in data:
                equipment = 3
            else:
                equipment = None

            has_lift = None
            if 'Výtah' in data:
                has_lift = 1

            internet = None
            if 'Internet' in data:
                has_lift = 1

            has_balcony = None
            if 'Balkón' in data:
                has_balcony = 1

            has_loggia = None
            if 'Lodžie' in data:
                has_loggia = 1

            has_cellar = None
            if 'Sklep' in data:
                has_cellar = 1

            has_garage = None
            if 'Garáž' in data:
                has_garage = 1

            has_parking = None
            if 'Parkování' in data:
                has_parking = 1

            heating = None # (?)
            if 'Elektrické' in data:
                heating = 1
            elif 'Ústřední' in data:
                heating = 2

            state = None
            if 'Kompletní' in data:
                state = 1
            elif 'Novostavba' in data:
                state = 2
            elif 'Velmi dobrý' in data:
                state = 3
            elif 'Dobrý' in data:
                state = 4
            elif 'Špatný' in data:
                state = 5

            index = data.index('PENB')
            energy_effeciency = data[index+1]
            print('Energie')
            print(energy_effeciency)
            index_2 = data.index('Podlaží')
            floor = data[index_2+7]
            print('Podlaží')
            print(floor)
            # places
            MHD = 0
            post_office = 0
            supermarket_grocery = 0
            atm = 0
            restaurant_pub = 0
            primary_school = 0
            kindergarten = 0
            pharmacy = 0
            playground_gym_pool_dist = 0
            places = soup.find('div', attrs={'class': 'row row-cols-lg-2 row-cols-1'})
            places = str(places.text)

            # if in places is MHD apod. 1 - 0
            if 'MHD' in places:
                MHD = 1
            if 'Pošta' in places:
                post_office = 1
            if 'Obchod' in places:
                supermarket_grocery = 1
            if 'Banka' in places:
                atm = 1
            if 'Restaurace' in places:
                restaurant_pub = 1
            if 'Škola' in places:
                primary_school = 1
            if 'Mateřská' in places:
                kindergarten = 1
            if 'Sportoviště' in places:
                playground_gym_pool_dist = 1
            if 'Hřiště' in places:
                playground_gym_pool_dist = 1
            if 'Lékárna' in places:
                pharmacy = 1

            # GPS


            out = {'header': header,  # text description of disposition e.g. 3 + kk
            'price': price,  # Celková cena
            'note': note,  # poznamka (k cene) sometimes can have valid info like
                # Při rychlém jednání možná sleva., včetně provize, včetně právního servisu, cena k jednání
            'usable_area': usable_area,  # Užitná plocha
            'floor_area': None,  # Plocha podlahová (?)
            'floor': floor,  # podlazie
            'energy_effeciency': energy_effeciency,  # Energetická náročnost (letters A-G) A=best, G=shitty
            'ownership': ownership,  # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
            'description': description,
            'long': None,
            'lat': None,
            'hash': None,

            # binary civic amenities (obcanska vybavenost binarne info) - done
            'MHD': MHD, # spíš MHD
            'train_station': None,
            'post_office': post_office,
            'atm': atm, # bankomat according to google translate :D
            'doctor': None,
            'vet': None,
            'primary_school': primary_school,
            'kindergarten': kindergarten,
            'supermarket_grocery': supermarket_grocery,
            'restaurant_pub': restaurant_pub,
            'playground_gym_pool': playground_gym_pool_dist, # or similar kind of leisure amenity probably OSM would be better
            'subway': None,
            'tram': None,
            # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
            'theatre_cinema': None,
            'pharmacy' : pharmacy,
            'internet' : internet,

            # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) - TODO (?)
            'bus_station_dist': None,
            'train_station_dist': None,
            'post_office_dist': None,
            'atm_dist': None,  # bankomat according to google translate :D
            'doctor_dist': None,
            'vet_dist': None,
            'primary_school_dist': None,
            'kindergarten_dist': None,
            'supermarket_grocery_dist': None,
            'restaurant_pub_dist': None,
            'playground_gym_pool_dist': None,  # or similar kind of leisure amenity probably OSM would be better
            'subway_dist': None,
            'tram_dist': None,
            # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
            'theatre_cinema_dist': None,

            # other - done
            'gas': None,  # Plyn
            'waste': None,  # Odpad
            'equipment': equipment,  # Vybavení
            'state': state,  # stav objektu e.g. po rekonstrukci/projekt etc  (10 states possible) see https://www.sreality.cz/hledani/byty
            'construction_type': construction_type,  # Stavba (3 states possible ) panel, cihla, ostatni
            'place': place,  # Umístění objektu
            'electricity': None,  # elektrina
            'heating': heating,  # topeni
            'transport': None,  # doprava
            'year_reconstruction': None,  # rok rekonstrukce
            'telecomunication': None,  # telekomunikace

            # binary info - done
            'has_lift': has_lift,  # Výtah: True, False
            'has_garage': has_garage,  # garaz
            'has_cellar': has_cellar,  # sklep presence or  m2 ???
            'no_barriers': None,  # ci je bezbarierovy bezbarierovy
            'has_loggia': has_loggia,  # lodzie m2
            'has_balcony': has_balcony,  # balkon
            'has_garden': None, # zahrada,
            'has_parking': has_parking,

            # additional info - sometimes
            'cellar_area': None, # plocha sklepu (if provided)
            'loggia_area': None,
            'balcony_area': None}

            print(out)
            return out
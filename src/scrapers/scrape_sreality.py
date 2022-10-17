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

import urllib3
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
        out = {  # basic sort of required info
            'header': None,  # text description of disposition e.g. 3 + kk
            'price': None,  # Celková cena
            'note': None,  # poznamka (k cene) sometimes can have valid info like
                # Při rychlém jednání možná sleva., včetně provize, včetně právního servisu, cena k jednání
            'usable_area': None,  # Užitná plocha
            'floor_area': None,  # Plocha podlahová
            'floor': None,  # podlazie
            'energy_effeciency': None,  # Energetická náročnost (letters A-G) A=best, G=shitty
            'ownership': None,  # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
            'long': None,
            'lat': None,
            'hash': None,

            # binary civic amenities (obcanska vybavenost binarne info)
            'bus_station': None,
            'train_station': None,
            'post_office': None,
            'atm': None, # bankomat according to google translate :D
            'doctor': None,
            'vet': None,
            'primary_school': None,
            'kindergarten': None,
            'supermarket_grocery': None,
            'restaurant_pub': None,
            'playground_gym_pool': None, # or similar kind of leisure amenity probably OSM would be better
            'subway': None,
            'tram': None,
            # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
            'theatre_cinema': None,

            # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti)

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

            # other
            'gas': None,  # Plyn
            'waste': None,  # Odpad:
            'equipment': None,  # Vybavení:
            'state': None,  # stav objektu e.g. po rekonstrukci/projekt etc  (10 states possible) see https://www.sreality.cz/hledani/byty
            'construction_type': None,  # Stavba (3 states possible ) panel, cihla, ostatni
            'place': None,  # Umístění objektu
            'electricity': None,  # elektrina
            'heating': None,  # topeni
            'transport': None,  # doprava
            'year_reconstruction': None,  # rok rekonstrukce
            'telecomunication': None,  # telekomunikace

            # binary info
            'has_lift': None,  # Výtah: True, False
            'has_garage': None,  # garaz
            'has_cellar': None,  # sklep presence or  m2 ???
            'no_barriers': None,  # ci je bezbarierovy bezbarierovy
            'has_loggia': None,  # lodzie m2
            'has_balcony': None,  # balkon
            'has_garden': None, # zahrada,
            'has_parking': None,

            # additional info
            'cellar_area': None, # plocha sklepu (if provided)
            'loggia_area': None,
            'balcony_area': None
        }
        if 'sreality' not in url:  # ensures correct link
            return {}
        else:
            # self._save_image(
            #                  'https://d18-a.sdn.cz/d_18/c_img_gU_o/YBJSIh.jpeg?fl=res,749,562,3|wrm,/watermark/sreality.png,10|shr,,20|jpg,90',
            #                  url)
            # # to get data
            try:
                driver.get(url)
                content = driver.page_source
                soup = BeautifulSoup(content)
                # scrape header
                header = soup.find('span', attrs={'class': 'name ng-binding'})
                header = str(header)
                if header is not None and header != "None":
                    header = header[header.find(">") + 1:]
                    header = header[:header.find("<")]
                    print(header)

                    # tabularni data
                    table_data = []
                    # for li in soup.findAll('li', href=True, attrs={'class': 'param ng_scope'}):
                    #     table_data.append(li)

                    table_data = soup.find('div', attrs={'class': 'params clear'})
                    table_data = table_data.text
                    table = re.split('\n\n\n\n\n|\n\n\n\n', table_data)
                    table[0] = table[0][2:]
                    print(str(table[0]))
                    for i in range(len(table)):
                        table[i] = str(table[i].replace("\xa0", ""))
                        table[i] = table[i].split(":\n\n")
                    slovnik = {}
                    for i in range(len(table)):
                        if len(table[i]) == 2:
                            slovnik[str(table[i][0])] = [str(table[i][1])]
                    dataframe = pd.DataFrame.from_dict(slovnik)
                    print(dataframe)

                    # scrape price
                    price = soup.find('span', attrs={'class': 'ng-binding ng-scope'})
                    price = str(price)
                    price = price[price.find(">") + 1:]
                    price = price[:price.find("Kč") + 2]

                    # scrape description
                    description = soup.find('div', attrs={'class': 'description ng-binding'})
                    retezec = str(description)
                    index_paragraph = retezec.find("<p>")
                    retezec = retezec[index_paragraph:]
                    retezec = retezec.replace("<p>", "")
                    retezec = retezec.replace("</p>", " ")
                    retezec = retezec.replace("</div>", "")
                    description = retezec
                    # del retezec
                    # price = a.find('div', attrs={'class': '_1vC4OE _2rQ-NK'})
                    # rating = a.find('div', attrs={'class': 'hGSR34 _2beYZw'})
                    # products.append(name.text)
                    # prices.append(price.text)
                    # ratings.append(rating.text)
                    # resp = http.request('GET', url)
                    # data = resp.data.decode('utf-8')
                    # print(description)
                # self._save_text(str(data), url)
            except:
                print("")
            test_dict = {'header': 'Prodej bytu 4+kk • 123 m² bez realitky',
                         'price': 17850000,
                         'plocha': 123,
                         'stav_objektu': 'Před rekonstrukcí',
                         'long': 45.1,
                         'lat': 15.5,
                         'hash': 'asdafqwf6sadasew6'}
            return test_dict

# if __name__ == "__main__":
#     s = SRealityScraper()
#     s.run(in_filename='prodej_links.txt', out_filename='prodej')
# projdi = SRealityScraper()
# print(projdi.scrape("https://sreality.cz/detail/prodej/byt/2+kk/praha-zizkov-seifertova/1535174476"))

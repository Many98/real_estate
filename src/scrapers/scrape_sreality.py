import pandas as pd
import time
import csv
import os
import re
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
            driver.get(url)
            content = driver.page_source
            time.sleep(0.5)
            soup = BeautifulSoup(content)
            time.sleep(0.5)
            error_page = soup.find('div', attrs={'class': 'error-page ng-scope'})
            if error_page is None:
                slovnik = {}
                # scrape header
                header = soup.find('span', attrs={'class': 'name ng-binding'})
                if header is not None:
                    header = str(header)
                    header = header[header.find(">") + 1:]
                    header = header[:header.find("<")]
                    slovnik["header"] = header
                else:
                    print("Header for this page does not exist.")


                # tabularni data
                table_data = soup.find('div', attrs={'class': 'params clear'})
                if table_data is not None:
                    table_data_str = str(table_data)
                    r = re.compile(r'\bICON-OK\b | \bICON-CROSS\b', flags=re.I | re.X)
                    crosses_and_ticks = r.findall(table_data_str)
                    yes_no = []
                    for item in crosses_and_ticks:
                        if item == "icon-ok":
                            yes_no.append("ano")
                        else:
                            yes_no.append("ne")
                    retezec = table_data.text
                    retezec = retezec.replace("\n", " ")
                    three_spaces_in_retezec = retezec.find("   ")
                    retezec = retezec.replace("\xa0", "")
                    while three_spaces_in_retezec >= 0:
                        retezec = retezec.replace("   ", "  ")
                        three_spaces_in_retezec = retezec.find("   ")

                    table = re.split('  ', retezec)
                    table = [value for value in table if value != ""]
                    size_of_table = len(table)
                    i = 0
                    j = 0
                    while i < size_of_table:
                        if i != size_of_table - 1:
                            if table[i][-1] == ":" and table[i+1][-1] == ":":
                                table.insert(i+1, yes_no[j])
                                j += 1
                                size_of_table = len(table)
                        elif i == size_of_table - 1 and table[i][-1] == ":":
                            table.append(yes_no[j])
                            size_of_table = len(table)
                        i+=1
                    for i in range(int(len(table)/2)):
                        i *= 2
                        slovnik[table[i]] = table[i+1]

                else:
                    print("No tabular data on this page")


                # scrape description
                description = soup.find('div', attrs={'class': 'description ng-binding'})
                if description is not None:
                    retezec = str(description)
                    index_paragraph = retezec.find("<p>")
                    retezec = retezec[index_paragraph:]
                    retezec = retezec.replace("<p>", "")
                    retezec = retezec.replace("</p>", " ")
                    retezec = retezec.replace("</div>", "")
                    retezec = retezec.replace("\xa0", "")
                    description = retezec
                    slovnik["description"] = description
                    del retezec
                else:
                    print("No description on this webpage.")


                # scrape position
                position = soup.find('a', attrs={'class': 'print'})
                if position is not None:
                    position = str(position)
                    index = position.find("x=")
                    position = position[index:]
                    index = position.find("style=")
                    position = position[:index]
                    position = re.split('&amp;', position)
                    slovnik["position"] = position
                else:
                    print("No position on this web page.")



                # public equipment scraping
                public_equipment = soup.find('preact', attrs={'data': 'publicEquipment'})
                time.sleep(0.5)
                if public_equipment is not None:
                    public_equipment = public_equipment.text
                    seznam = public_equipment.split(")")
                    seznam = [item + ")" for item in seznam]
                    seznam = seznam[:-1]
                    for i in range(len(seznam)):
                        seznam[i] = seznam[i].split(":")
                    retezec = seznam[0][0]
                    words = re.findall('[A-Z][^A-Z]*', retezec)
                    seznam[0][0] = words[-1]
                    for i in range(len(seznam)):
                        slovnik[seznam[i][0]] = seznam[i][1]






            test_dict = {'header': 'Prodej bytu 4+kk • 123 m² bez realitky',
                         'price': 17850000,
                         'plocha': 123,
                         'stav_objektu': 'Před rekonstrukcí',
                         'long': 45.1,
                         'lat': 15.5,
                         'hash': 'asdafqwf6sadasew6'}
            return test_dict

import pandas as pd
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
                    slovnik = {}
                    for i in range(int(len(table)/2)):
                        i *= 2
                        slovnik[table[i]] = table[i+1]

                    # scrape description
                    description = soup.find('div', attrs={'class': 'description ng-binding'})
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


                    # scrape position
                    position = soup.find('a', attrs={'class': 'print'})
                    position = str(position)
                    index = position.find("x=")
                    position = position[index:]
                    index = position.find("style=")
                    position = position[:index]
                    position = re.split('&amp;', position)
                    slovnik["position"] = position



                    # public equipment scraping
                    public_equipment = soup.find('preact', attrs={'data': 'publicEquipment'})
                    public_equipment = public_equipment.text


                    #price = a.find('div', attrs={'class': '_1vC4OE _2rQ-NK'})
                    #rating = a.find('div', attrs={'class': 'hGSR34 _2beYZw'})
                    #products.append(name.text)
                    #prices.append(price.text)
                    #ratings.append(rating.text)
                    #resp = http.request('GET', url)
                    #data = resp.data.decode('utf-8')
                    # print(description)
                #self._save_text(str(data), url)
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
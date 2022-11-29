import time
import os
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from datetime import datetime
from bs4 import BeautifulSoup
import re

from scrapers.BaseScraper import BaseScraper


class SRealityScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'sreality') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> dict:
        """
            Here comes custom implementation for sreality.cz
            E.g.  method must be able to scrape all relevant data from urls like
            https://sreality.cz/detail/prodej/byt/1+kk/praha-holesovice-veletrzni/3827836492
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
        out = {
            'header': None,
            # text description of disposition e.g. 3 + kk
            'price': None,  # Celková cena
            'note': None,
            # poznamka (k cene) sometimes can have valid info like
            # Při rychlém jednání možná sleva., včetně provize, včetně právního servisu, cena k jednání
            'usable_area': None,  # Užitná plocha
            'floor_area': None,  # Plocha podlahová
            'floor': None,  # podlazie
            'energy_effeciency': None,
            # Energetická náročnost (letters A-G) A=best, G=shitty
            'ownership': None,
            # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
            'description': None,
            'long': None,
            'lat': None,
            'hash': None,

            # other - done
            'gas': None,  # Plyn
            'waste': None,  # Odpad
            'equipment': None,  # Vybavení
            'state': None,
            # stav objektu e.g. po rekonstrukci/projekt etc  (10 states possible) see https://www.sreality.cz/hledani/byty
            'construction_type': None,
            # Stavba (3 states possible ) panel, cihla, ostatni
            'place': None,  # Umístění objektu
            'electricity': None,  # elektrina
            'heating': None,  # topeni
            'transport': None,  # doprava
            'year_reconstruction': None,  # rok rekonstrukce
            'telecomunication': None,  # telekomunikace

            # binary info - done
            'has_lift': None,  # Výtah: True, False
            'has_garage': None,  # garaz
            'has_cellar': None,  # sklep presence
            'no_barriers': None,  # ci je bezbarierovy bezbarierovy
            'has_loggia': None,  # lodzie
            'has_balcony': None,  # balkon
            'has_garden': None,  # zahrada
            'has_parking': None,

            # what has b reality in addition
            'tags': None,
            'disposition': None,

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

        if 'sreality' not in url:  # ensures correct link
            return {}
        else:
            self.cookie_added = False
            driver.get(url)
            # this adds cookie to get rid of cookie constent popup
            # because it was causing unexpected results when loading elements with selenium
            if not self.cookie_added:
                self.add_consent_cookie(driver)
            time.sleep(2)
            content = driver.page_source
            soup = BeautifulSoup(content, features="lxml")
            error_page = soup.find('div', attrs={'class': 'error-page ng-scope'})
            if error_page is None:
                slovnik = {}
                # scrape header
                try:
                    #time.sleep(1)
                    header = soup.find('span', attrs={'class': 'name ng-binding'})
                    if header is None:
                        raise Exception('header not found')
                except:
                    try:
                        header = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                 '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[4]/h1/span/span[1]')))
                    except:
                        print('No header present, link is probably expired')
                        with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                            f.write(url + '\n')
                        return {'status': 'expired'}
                if header is not None:
                    slovnik["header"] = header.text
                else:
                    print("Header for this page does not exist.")

                # price
                try:
                    time.sleep(1)
                    price = soup.find('span', attrs={'class': 'norm-price ng-binding'})
                    if price is None:
                        raise Exception('header not found')
                except:
                    try:
                        #time.sleep(5)
                        price = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[4]/span/span[2]')))
                    except:
                        price = None
                        with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                            f.write(url + ' price ' + '\n')
                if price is not None:
                    slovnik["price"] = price.text.replace("\xa0", "")
                else:
                    print("Price for this page does not exist.")

                # tabular data
                try:
                    #time.sleep(1)
                    table_data = soup.find('div', attrs={'class': 'params clear'})
                    if table_data is None:
                        raise Exception('table data not found')
                except:
                    try:
                        time.sleep(15)
                        content = driver.page_source
                        soup = BeautifulSoup(content, features="lxml")
                        table_data = soup.find('div', attrs={'class': 'params clear'})
                        if table_data is None:
                            raise Exception('table data not found after wait')
                    except:
                        try:
                            table_data = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                         '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[7]')))
                        except:
                            table_data = None
                            with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                                f.write(url + ' tabular_data ' + '\n')

                if table_data is not None and table_data.text:
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
                    while i < size_of_table and j < len(yes_no):
                        if i < size_of_table - 1:
                            if table[i][-1] == ":" and table[i + 1][-1] == ":":
                                table.insert(i + 1, yes_no[j])
                                j += 1
                                size_of_table = len(table)
                        elif i == size_of_table - 1 and table[i][-1] == ":":
                            table.append(yes_no[j])
                            size_of_table = len(table)
                        i += 1
                    for i in range(int(len(table) / 2)):
                        i *= 2
                        slovnik[table[i]] = table[i + 1]

                else:
                    print("No tabular data on this page")

                # scrape description
                try:
                    #time.sleep(1)
                    description = soup.find('div', attrs={'class': 'description ng-binding'})
                    if description is None or description.text == '':
                        raise Exception('description not found')
                except:
                    try:
                        time.sleep(5)
                        description = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH,
                                                                                                      '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[6]')))
                    except:
                        description = None
                        with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                            f.write(url + ' description ' + '\n')
                if description is not None:
                    slovnik["description"] = description.text.replace("\xa0", "")
                else:
                    print("No description on this webpage.")

                # scrape position
                try:
                    #time.sleep(1)
                    position = soup.find('a', attrs={'class': 'print'})
                    if position is None:
                        raise Exception('position not found')
                except:
                    try:
                        position = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                   '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[2]/div[2]/div/a')))  # //*[@id="s-map"]/div[2]/div/a
                        position = position.get_attribute('href')
                    except:
                        position = None
                        with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                            f.write(url + ' position ' + '\n')

                if position is not None:
                    position = str(position)
                    position = position[position.find("x="):]
                    position = position[:position.find("style=")].replace('amp', '').replace(';', '')
                    slovnik["position"] = position.split('&')[:-1]
                else:
                    print("No position on this web page.")

                # public equipment scraping
                try:
                    #time.sleep(1)
                    public_equipment = soup.find('preact', attrs={'data': 'publicEquipment'})
                    if public_equipment is None:
                        raise Exception('public equipment not found')
                except:
                    try:
                        time.sleep(15)
                        content = driver.page_source
                        soup = BeautifulSoup(content, features="lxml")
                        public_equipment = soup.find('preact', attrs={'data': 'publicEquipment'})
                        if public_equipment is None:
                            raise Exception('table data not found after wait')
                    except:
                        try:
                            # time.sleep(5)
                            public_equipment = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                               '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/preact[1]/div/ul')))
                        except:
                            try:
                                public_equipment = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                                                   '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/preact[1]')))
                            except:
                                public_equipment = None
                                with open(os.path.join('../', 'data', 'weird_links.txt'), 'a') as f:
                                    f.write(url + ' public_equip ' + '\n')

                if public_equipment is not None and public_equipment.text:
                    public_equipment = public_equipment.text
                    seznam = public_equipment.split("m)")
                    seznam = [item + ")" for item in seznam]
                    seznam = seznam[:-1]
                    for i in range(len(seznam)):
                        seznam[i] = seznam[i].split(":")
                    retezec = seznam[0][0]
                    words = re.findall('[A-Z][^A-Z]*', retezec)
                    seznam[0][0] = words[-1]
                    for i in range(len(seznam)):
                        slovnik[seznam[i][0]] = seznam[i][1]

                # filling out dictionary from slovnik
                if "header" in slovnik:
                    out["header"] = slovnik["header"]
                    del slovnik["header"]
                if "Celková cena:" in slovnik:
                    out["price"] = slovnik["Celková cena:"]
                    del slovnik["Celková cena:"]
                elif "Cena:" in slovnik:
                    out["price"] = slovnik["Cena:"]
                    del slovnik["Cena:"]
                elif "price" in slovnik:
                    out["price"] = slovnik["price"]
                    del slovnik["price"]
                if "Poznámka k ceně:" in slovnik:
                    out["note"] = slovnik["Poznámka k ceně:"]
                    del slovnik["Poznámka k ceně:"]
                if "Užitná plocha:" in slovnik:
                    out["usable_area"] = slovnik["Užitná plocha:"]
                    del slovnik["Užitná plocha:"]
                if "Podlahová plocha:" in slovnik:
                    out["floor_area"] = slovnik["Podlahová plocha:"]
                    del slovnik["Podlahová plocha:"]
                if "Podlaží:" in slovnik:
                    out["floor"] = slovnik["Podlaží:"]
                    del slovnik["Podlaží:"]
                if "Energetická náročnost:" in slovnik:
                    out["energy_effeciency"] = slovnik["Energetická náročnost:"]
                    del slovnik["Energetická náročnost:"]
                if "Energetická náročnost budovy:" in slovnik:
                    out["energy_effeciency"] = slovnik["Energetická náročnost budovy:"]
                    del slovnik["Energetická náročnost budovy:"]
                if "Vlastnictví:" in slovnik:
                    out["ownership"] = slovnik["Vlastnictví:"]
                    del slovnik["Vlastnictví:"]
                if "position" in slovnik:
                    out["long"] = slovnik["position"][0]
                    out["lat"] = slovnik["position"][1]
                    del slovnik["position"]
                if "description" in slovnik:
                    out["description"] = slovnik["description"]
                    del slovnik["description"]
                if "Bus MHD" in slovnik:
                    out['bus_station_dist'] = slovnik["Bus MHD"]
                    del slovnik["Bus MHD"]
                if "Bus" in slovnik:
                    out['bus_station_dist'] = slovnik["Bus"]
                    del slovnik["Bus"]
                if "Vlak" in slovnik:
                    # out["train_station"] = "yes"
                    out["train_station_dist"] = slovnik["Vlak"]
                    del slovnik["Vlak"]
                if "Pošta" in slovnik:
                    # out["post_office"] = "yes"
                    out["post_office_dist"] = slovnik["Pošta"]
                    del slovnik["Pošta"]
                if "Bankomat" in slovnik:
                    # out["atm"] = "yes"
                    out["atm_dist"] = slovnik["Bankomat"]
                    del slovnik["Bankomat"]
                if "Lékař" in slovnik:
                    # out["doctor"] = "yes"
                    out["doctor_dist"] = slovnik["Lékař"]
                    del slovnik["Lékař"]
                if "Lékárna" in slovnik:
                    out["pharmacy_dist"] = slovnik["Lékárna"]
                    del slovnik["Lékárna"]
                if "Veterinář" in slovnik:
                    # out["vet"] = "yes"
                    out["vet_dist"] = slovnik["Veterinář"]
                    del slovnik["Veterinář"]
                if "Škola" in slovnik:
                    # out["primary_school"] = "yes"
                    out["primary_school_dist"] = slovnik["Škola"]
                    del slovnik["Škola"]
                if "Školka" in slovnik:
                    # out["kindergarten"] = "yes"
                    out["kindergarten_dist"] = slovnik["Školka"]
                    del slovnik["Školka"]
                if "Večerka" in slovnik:
                    # out["supermarket_grocery"] = "yes"
                    out["supermarket_grocery_dist"] = slovnik["Večerka"]
                    del slovnik["Večerka"]
                elif "Obchod" in slovnik:
                    # out["supermarket_grocery"] = "yes"
                    out["supermarket_grocery_dist"] = slovnik["Obchod"]
                    del slovnik["Obchod"]
                if "Restaurace" in slovnik:
                    # out["restaurant_pub"] = "yes"
                    out["restaurant_pub_dist"] = slovnik["Restaurace"]
                    del slovnik["Restaurace"]
                elif "Hospoda" in slovnik:
                    # out["retaurant_pub"] = "yes"
                    out["restaurant_pub_dist"] = slovnik["Hospoda"]
                    del slovnik["Hospoda"]
                if "Hřiště" in slovnik:
                    # out['playground_gym_pool'] = "yes"
                    out['playground_dist'] = slovnik["Hřiště"]
                    del slovnik["Hřiště"]
                if "Sportoviště" in slovnik:
                    # out['sports_field'] = "yes"
                    out['sports_field_dist'] = slovnik["Sportoviště"]
                    del slovnik["Sportoviště"]
                if "Hřiště" in slovnik:
                    # out["playground"] = "yes"
                    out["playground_dist"] = slovnik["Hřiště"]
                    del slovnik["Hřiště"]
                if "Metro" in slovnik:
                    # out["subway"] = "yes"
                    out["subway_station_dist"] = slovnik["Metro"]
                    del slovnik["Metro"]
                if "Tram" in slovnik:
                    # out["tram"] = "yes"
                    out["tram_station_dist"] = slovnik["Tram"]
                    del slovnik["Tram"]
                if "Divadlo" in slovnik:
                    # out["theatre_cinema"] = "yes"
                    out["theatre_cinema_dist"] = slovnik["Divadlo"]
                    del slovnik["Divadlo"]
                elif "Kino" in slovnik:
                    # out["theatre_cinema"] = "yes"
                    out["theatre_cinema_dist"] = slovnik["Kino"]
                    del slovnik["Kino"]
                if "Plyn:" in slovnik:
                    out["gas"] = slovnik["Plyn:"]
                    del slovnik["Plyn:"]
                if "Odpad:" in slovnik:
                    out["waste"] = slovnik["Odpad:"]
                    del slovnik["Odpad:"]
                if "Vybavení:" in slovnik:
                    out["equipment"] = slovnik["Vybavení:"]
                    del slovnik["Vybavení:"]
                if "Stav objektu:" in slovnik:
                    out["state"] = slovnik["Stav objektu:"]
                    del slovnik["Stav objektu:"]
                if "Stavba:" in slovnik:
                    out["construction_type"] = slovnik["Stavba:"]
                    del slovnik["Stavba:"]
                if "Umístění objektu:" in slovnik:
                    out["place"] = slovnik["Umístění objektu:"]
                    del slovnik["Umístění objektu:"]
                if "Elektřina:" in slovnik:
                    out["electricity"] = slovnik["Elektřina:"]
                    del slovnik["Elektřina:"]
                if "Topení:" in slovnik:
                    out["heating"] = slovnik["Topení:"]
                    del slovnik["Topení:"]
                if "Doprava:" in slovnik:
                    out["transport"] = slovnik["Doprava:"]
                    del slovnik["Doprava:"]
                if "Rok rekonstrukce:" in slovnik:
                    out["year_reconstruction"] = slovnik["Rok rekonstrukce:"]
                    del slovnik["Rok rekonstrukce:"]
                if "Telekomunikace:" in slovnik:
                    out["telecomunication"] = slovnik["Telekomunikace:"]
                    del slovnik["Telekomunikace:"]
                if "Výtah:" in slovnik:
                    out["has_lift"] = slovnik["Výtah:"]
                    del slovnik["Výtah:"]
                if "Garáž:" in slovnik:
                    out["has_garage"] = slovnik["Garáž:"]
                    del slovnik["Garáž:"]
                if "Sklep:" in slovnik:
                    out["has_cellar"] = slovnik["Sklep:"]
                    del slovnik["Sklep:"]
                if "Bezbariérový přístup:" in slovnik:
                    out["no_barriers"] = slovnik["Bezbariérový přístup:"]
                    del slovnik["Bezbariérový přístup:"]
                elif "Bezbariérový:" in slovnik:
                    out["no_barriers"] = slovnik["Bezbariérový:"]
                    del slovnik["Bezbariérový:"]
                if "Lodžie:" in slovnik:
                    out["has_loggia"] = slovnik["Lodžie:"]
                    del slovnik["Lodžie:"]
                if "Balkón:" in slovnik:
                    out["has_balcony"] = slovnik["Balkón:"]
                    del slovnik["Balkón:"]
                if "Zahrada:" in slovnik:
                    out["has_garden"] = slovnik["Zahrada:"]
                    del slovnik["Zahrada:"]
                if "Parkování:" in slovnik:
                    out["has_parking"] = slovnik["Parkování:"]
                    del slovnik["Parkování:"]
                if "Lékárna" in slovnik:
                    out["pharmacy"] = "yes"
                    out["pharmacy_dist"] = slovnik["Lékárna"]
                    del slovnik["Lékárna"]
                if out["description"] is None:
                    out["description"] = ""

                for k, v in slovnik.items():
                    out["description"] += ' ' + k + ":" + v + ". "
                del slovnik

                return out
            else:
                return {'status': 'expired'}

    def add_consent_cookie(self, driver: WebDriver) -> None:
        try:
            driver.add_cookie({'domain': '.sreality.cz',
                               'expiry': 1702034432,
                               'httpOnly': False,
                               'name': 'euconsent-v2',
                               'path': '/',
                               'sameSite': 'None',
                               'secure': True,
                               'value': 'CPiEukAPiEukAD3ACBCSCoCsAP_AAEPAAATIIDoBhCokBSFCAGpYIIMAAAAHxxAAYCACABAAgAABABIAIAQAAAAQAAQgBAAAABQAIAIAAAAACEAAAAAAAAAAAQAAAAAAAAAAIQAAAAAAAiBAAAAAAABAAAAAAABAQAAAgAAAAAIAQAAAAAEAgAAAAAAAAAAAAAAAAQgAAAAAAAAAAAganAlAAWABUAC4AGQAQAAyABoADmAIgAigBMACeAFUAMQAfgBCQCIAIkARwAnABSgCxAGWAM0AdwA_QCEAEWALQAXUAwIBrAD5AJtAWoAvMBkgDSgGpgAA.YAAAAAAAAAAA'})
            self.cookie_added = True
        except:
            pass

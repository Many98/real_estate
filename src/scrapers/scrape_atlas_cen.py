import pandas as pd
import csv
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from datetime import date
from dateutil.relativedelta import relativedelta

from scrapers.BaseScraper import BaseScraper


class AtlasCenScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'atlas_cen') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> list:
        """Here comes custom implementation for atlas cen
        """
        data = []

        page = 1
        # for prague last 6 months from now
        url = f'https://www.reas.cz/atlas?bounds=49.88755653624285%2C13.974609375000002%2C50.23183414485175%2C14.956512451171877&filters=%7B%22types%22%3A%5B%22flat%22%5D%2C%22sort%22%3A%22sales_with_photos%22%2C%22date%22%3A%7B%22selectedOption%22%3A%22six_months%22%2C%22soldAfter%22%3A%22{str(date.today() + relativedelta(months=-6))}%22%7D%7D&search=%7B%22text%22%3A%22Praha%22%7D&scrollPos=&listPage={page}&listPerPage=30'
        pagination = True
        driver.get(url)
        ele = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH,
                                    '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/div/div/p')))
        time.sleep(3)
        page_count = int(re.sub('[^0-9]', '', ele.text)) // 30 + 1
        stop = 0
        while pagination and page <= page_count:  # page <= page_count
            time.sleep(1)
            x = int(50 * page / page_count) + 1
            print(f'Page {page}/{page_count} [{u"█" * x}{"." * (51 - x)}]', end="\r", flush=True)  # 30 results per page is set in query
            try:
                ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
                when = WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions).until(EC.presence_of_all_elements_located((By.XPATH,
                                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/div[1]/p')))
                area = WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions).until(EC.presence_of_all_elements_located((By.XPATH,
                                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a/div/div[1]/div[1]/p[1]')))
                price = WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions).until(EC.presence_of_all_elements_located((By.XPATH,
                                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a/div/div[2]')))
                href = WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions).until(EC.presence_of_all_elements_located((By.XPATH,
                                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a')))
            except:
                # probably pop-up screen appeared
                self.close_pop_up(driver)
                time.sleep(2)
                if stop == 3:
                    driver.refresh()
                elif stop > 3:
                    raise Exception('Unknown page error')
                stop += 1
                continue
            stop = 0
            time.sleep(1)

            dynamic_button = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/button')))
            if dynamic_button.is_displayed():
                dynamic_button.click()

            #estate_items = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
            #                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div[*]')))
            #estate_items2 = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
            #                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]')))
            #data += [{'text': i.text, 'href': i.find_element_by_css_selector('a').get_attribute('href')} for i in
            #         estate_items2]
            existing_hrefs = self.get_existing_links()
            try:
                data = [{'when': i.text, 'href': j.get_attribute('href'), 'area': k.text, 'price': l.text} for i, j, k, l
                        in zip(when, href, area, price)]
                data_out = [i for i in data if i['href'] not in existing_hrefs]
                ll = [list(map(float, re.findall("\d+\.\d+", i['href'][:110]))) for i in data_out]
                for i, j in zip(data_out, ll):
                    i.update({'long': j[1], 'lat': j[0]})
                # finally filter already existing data according to href attribute (kind of unique identifier)
                self.data = data_out
                if len(data_out) != len(data):
                    print('Some records already exists', end="\r", flush=True)
            except:
                print('something went wrong')
                self.close_pop_up(driver)
                time.sleep(3)
                if stop == 3:
                    driver.refresh()
                elif stop > 3:
                    raise Exception('Unknown page error')
                stop += 1
                continue

            try:
                #next_page_elem = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH,
                #        '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div[1]/button[*]')))
                next_page_elem = WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.XPATH,
                    '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div[1]/button[*]')))
                try:
                    next_page_elem[-1].click()
                except:
                    self.close_pop_up(driver)  # close pop-up
                    try:
                        next_page_elem[-1].click()  # go to next page
                    except:
                        pagination = False
                        continue
            except:
                pagination = False
                continue

            """
            try:
                url = url.replace(f'listPage={page}', f'listPage={page+1}')
                driver.get(url)
            except:
                print('next page not found')
            """
            self._export_tabular_data()
            self.data = []
            page += 1

        return data

    def close_pop_up(self, driver: WebDriver) -> None:
        try:
            pop_up_close = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH,
                                                                            '/html/body/div[3]/div/i')))
            pop_up_close.click()
        except:
            pass

    def get_existing_links(self) -> list:
        '''reads the file of existing data'''
        path = os.path.join('../', 'data', f"_{self.name}_scraped.csv")  # TODO should be more robust
        if os.path.exists(path):
            data = pd.read_csv(path)

            return data['href'].tolist()
        else:
            return []


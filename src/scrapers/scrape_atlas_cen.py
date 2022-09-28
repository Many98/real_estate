import pandas as pd
import csv
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from scrapers.BaseScraper import BaseScraper


class AtlasCenScraper(BaseScraper):
    def __init__(self, delay: float = 0.5, name: str = 'atlas_cen') -> None:
        super().__init__(name, delay)

    def scrape(self, driver: WebDriver, url: str) -> list:
        """Here comes custom implementation for atlas cen
        """
        # for prague last year
        url = 'https://www.reas.cz/atlas?bounds=49.922051297763346%2C13.96018981933594%2C50.26608218923894%2C14.942092895507814&filters=%7B%22types%22%3A%5B%22flat%22%5D%2C%22sort%22%3A%22sales_with_photos%22%2C%22date%22%3A%7B%22selectedOption%22%3A%22last_year%22%2C%22soldAfter%22%3A%222021-9-28%22%7D%7D&search=%7B%22text%22%3A%22%22%7D&scrollPos=&listPage=1&listPerPage=30'
        data = []


        page = 1
        pagination = True
        driver.get(url)
        while pagination:
            time.sleep(5)
            when = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/div[1]/p')))
            area = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a/div/div[1]/div[1]/p[1]')))
            price = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a/div/div[2]')))
            href = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]/a')))

            time.sleep(2)

            dynamic_button = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH,
                            '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/button')))
            dynamic_button.click()

            #estate_items = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
            #                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div[*]')))
            #estate_items2 = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH,
            #                '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div[*]')))
            #data += [{'text': i.text, 'href': i.find_element_by_css_selector('a').get_attribute('href')} for i in
            #         estate_items2]
            data += [{'when': i.text, 'href': j.get_attribute('href'), 'area': k.text, 'price': l.text} for i, j, k, l
                     in zip(when, href, area, price)]

            try:
                #next_page_elem = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH,
                #        '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div[1]/button[*]')))
                next_page_elem = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH,
                    '//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[2]/div[1]/button[*]')))
                next_page_elem[-1].click()
            except:
                pagination = False
                continue

            page += 1

        return data

import time

import requests
from bs4 import BeautifulSoup
from scrapers.BaseKindOfCrawler import BaseKindOfCrawler
from tqdm import tqdm

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


class KindOfCrawlerForBezRealitky(BaseKindOfCrawler):

    def __init__(self,
                 out_filename: str,
                 url: str):

        super().__init__(out_filename, url)
        assert 'https://www.bezrealitky.cz' in self.main_url, 'Probably wrong url'

    def crawl(self) -> None:
        """Custom crawling logic for bezrealitky.cz"""
        print(f'Crawling breality from url: {self.main_url}')

        try:
            pagination = True
            i = 0
            page = 1
            # create soup object of html of main url
            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get(self.main_url)
            time.sleep(5)
            soup = BeautifulSoup(driver.page_source, 'lxml')
            next_page_elem = soup.find_all("a", {"class": "page-link"})
            href = [i.get('href') for i in next_page_elem if f'page={str(page + 1)}' in i.get('href')]
            href = href[0].replace(f'page={page+1}', f'page={page}')
            stop = 0
            while pagination:

                # get all links of apts
                ap_list_elem = soup.find_all('a')

                links = set([i.get("href") for i in ap_list_elem if 'nemovitosti-byty-domy' in i.get('href')])

                if not links:
                    try:
                        hrefs = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH,
                                                                                                         '//*[@id="__next"]/main/section/div/div[2]/div/div[4]/section/article[*]/div[1]/a')))
                        h = [i.get_attribute("href") for i in hrefs]
                        links = set([i for i in h if 'nemovitosti-byty-domy' in i])
                        if not links:
                            print(f'No links found on this page')
                            raise Exception('links not present')
                    except:
                        stop += 1
                        if stop == 3:  # after 4 unsuccessful fetching break
                            break
                # for each link, get the url from href
                for link_url in tqdm(links, desc=f'Fetching valid urls on page {page}'):

                    # if the link exists in the database, ignore
                    if link_url in self.existing_links:
                        print(f'Link {link_url} exists!')

                    # else: 1. add to database; 2. append to new apts list; 3. append to existing links list
                    else:
                        self.reality_links.append(link_url)
                        self.append_to_txt(link_url)
                        self.existing_links.append(link_url)
                        i += 1
                try:
                    href = href.replace(f'page={page}', f'page={page + 1}')
                    driver.get(href)
                    time.sleep(4.5)
                    soup = BeautifulSoup(driver.page_source, 'lxml')
                except:
                    pagination = False
                    continue
                page += 1
            # print number of new found apts
            print(f'Found {i} apartments')

        except:
            print('Something went wrong :(')

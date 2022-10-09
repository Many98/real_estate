from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from scrapers.BaseKindOfCrawler import BaseKindOfCrawler
from tqdm import tqdm


class KindOfCrawlerForSReality(BaseKindOfCrawler):

    def __init__(self,
                 out_filename: str,
                 url: str):

        super().__init__(out_filename, url)
        assert 'https://www.sreality.cz/' in self.main_url, 'Probably wrong url'

    def crawl(self) -> None:
        """ custom crawling logic for sreality.cz"""
        try:
            # options for the web driver
            options = Options()
            #options.add_argument('headless')
            #options.add_argument('--disable-infobars')
            #options.add_argument('profile.default_content_setting_values.cookies=2')
            #options.add_argument('--disable-dev-shm-usage')
            #options.add_argument('--no-sandbox')
            #options.add_argument('--remote-debugging-port=9222')

            print('Running webdriver...')

            # instantiate webdriver; install webdriver according to current chrome version
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
            print(f'Crawling sreality from url: {self.main_url}')

            # open main url in chrome
            try:
                driver.get(self.main_url)
            except:
                print(f'Main url {self.main_url} not found')
            time.sleep(1.5)

            pagination = True
            i = 0
            page = 1
            while pagination:

                # get html from the page
                page_soup = BeautifulSoup(driver.page_source, 'lxml')

                # quit the driver
                #driver.quit()

                # select all links with apts
                title_elem = page_soup.select('a.title')

                # for each apt link get href
                for link in tqdm(title_elem, desc=f'Fetching valid urls on page {page}'):
                    link_url = 'https://sreality.cz' + link.get('href')

                    # if the link exists in the database, ignore
                    if link_url in self.existing_links:
                        print(f'Link {link_url} exists!')

                    # else: 1. add to database; 2. append to new apts list; 3. append to existing links list
                    else:
                        try:  # TODO implement faster way of retrieving urls without checking of validity
                            driver.get(link_url)
                        except:
                            print(f'Not valid advertisement url {link_url}')
                        time.sleep(3)
                        soup = BeautifulSoup(driver.page_source, 'lxml')
                        #driver.quit()

                        if 'Je mi líto, inzerát neexistuje.' not in str(soup.body):
                            self.reality_links.append(link_url)
                            self.append_to_txt(link_url)
                            self.existing_links.append(link_url)
                            i += 1

                        else:

                            print(f'Link {link_url} is non-existing! SHAME!')

                try:
                    next_page_elem = page_soup.find("a", {"class": "btn-paging-pn icof icon-arr-right paging-next"})
                    driver.get('https://www.sreality.cz/' + next_page_elem.get('href'))
                except:
                    try:
                        next_page_elem = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH,
                                        '//*[@id="page-layout"]/div[2]/div[3]/div[3]/div/div/div/div/div[3]/div/div[25]/ul[2]/li[7]/a')))
                        driver.get(next_page_elem.get_attribute('href'))
                    except:
                        pagination = False
                        continue

                time.sleep(1.5)
                page += 1
            print(f'Found {i} apartments')
        except:

            print('Something went wrong :(')



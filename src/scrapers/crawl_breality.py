import requests
from bs4 import BeautifulSoup
from scrapers.BaseKindOfCrawler import BaseKindOfCrawler
from tqdm import tqdm


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
            soup = BeautifulSoup(requests.get(self.main_url).content, 'lxml')
            while pagination:

                # get all links of apts
                ap_list_elem = soup.find_all('a')

                links = set([i.get("href") for i in ap_list_elem if 'nemovitosti-byty-domy' in i.get('href')])
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
                    next_page_elem = soup.find_all("a", {"class": "page-link"})
                    href = [i.get('href') for i in next_page_elem if f'page={str(page+1)}' in i.get('href')]
                    # create soup object of html of main url
                    soup = BeautifulSoup(requests.get(href[0]).content, 'lxml') # if href is empty it will raise exception which is wanted
                except:
                    pagination = False
                    continue
                page += 1
            # print number of new found apts
            print(f'Found {i} apartments')

        except:
            print('Something went wrong :(')

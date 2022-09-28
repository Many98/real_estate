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
            # create soup object of html of main url
            soup = BeautifulSoup(requests.get(self.main_url).content, 'lxml')
            # get all links of apts
            ap_list_elem = soup.find_all('a')

            i = 0

            # for each link, get the url from href
            for link in ap_list_elem:
                link_url = link.get("href")
                if 'nemovitosti-byty-domy' in link_url:

                    # if the link exists in the database, ignore
                    if link_url in self.existing_links:
                        print(f'Link {link_url} exists!')

                    # else: 1. add to database; 2. append to new apts list; 3. append to existing links list
                    else:
                        self.reality_links.append(link_url)
                        self.append_to_txt(link_url)
                        self.existing_links.append(link_url)
                        i += 1

            # print number of new found apts
            print(f'Found {i} apartments')

        except:
            print('URL not provided.')

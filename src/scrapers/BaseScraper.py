from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import os
import glob
from selenium.webdriver.remote.webdriver import WebDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import base64

from tqdm import tqdm


class BaseScraper(ABC):
    def __init__(self, name: str, delay: float = 0.5) -> None:
        super().__init__()
        self.delay = delay
        self.name = name
        self.data = []  # storage of dicts of data
        self.prepared_links = []
        self.new_scraped_links = []
        self.scraped_links = []

    @abstractmethod   
    def scrape(self, driver: WebDriver, url: str) -> Union[dict, list]:
        """Loads page and scrape data from loaded web pages and returns dict or list of data"""
        pass

    def _export_tabular_data(self, out_filename: str = '', **kwargs) -> None:
        """method to export(append) tabular data to .csv file"""
        path = os.path.join('../', 'data', f"{out_filename}_{self.name}_scraped.csv")
        if self.data:
            df_to_be_written = pd.DataFrame(self.data)
            df_to_be_written.to_csv(path, mode='a', index=False, header=not os.path.exists(path))

        print(f'New data appended successfully in {path}!', end="\r", flush=True)

    def _update_scrapped_links(self, **kwargs) -> None:
        """update `already_scraped_links.txt` file"""
        if self.new_scraped_links and not kwargs.get('inference', False):
            with open(os.path.join('../', 'data', 'already_scraped_links.txt'), 'a') as f:
                f.writelines(['\n' + i for i in self.new_scraped_links])

    def _save_image(self, img_url: str, web_url: str) -> None:
        """Method to save image from `img_url`"""
        filename = web_url.strip().split('/')[-1].strip()
        dir_path = self._create_dir(web_url)
        try:
            if not os.path.exists(os.path.join(dir_path, filename)):
                response = requests.get(img_url, stream=True)
                with open(os.path.join(dir_path, filename), 'wb') as f:
                    f.write(response.content)
        except:
            print(f'Something went wrong with image downloading: {img_url}')

    def _save_text(self, scraped_text: str, web_url: str) -> None:
        """Method to save `scraped_text`"""
        dir_path = self._create_dir(web_url)
        if not os.path.exists(os.path.join(dir_path, 'text.txt')):
            with open(os.path.join(dir_path, 'text.txt'), 'w') as f:
                f.writelines(scraped_text)

    def _create_dir(self, web_url: str) -> str:
        """Method created directory for storing images and text for particular web_url"""
        dirname = self._hash(web_url)
        dir_ = os.path.join('../', 'data', 'blob', dirname)
        os.makedirs(dir_, exist_ok=True)

        return dir_

    def _hash(self, web_url: str) -> str:
        return base64.b64encode(web_url.encode("utf-8")).decode("utf-8")

    def _prepare(self, in_filename: str = '', **kwargs) -> None:
        """read .txt file with links"""
        if in_filename != '':
            with open(os.path.join('../', 'data', in_filename), 'r') as f:
                self.prepared_links = [line.rstrip() for line in f.readlines()]

            if not kwargs.get('inference', False):
                if not glob.glob(os.path.join('../', 'data', 'already_scraped_links.txt')):
                    with open(os.path.join('../', 'data', 'already_scraped_links.txt'), 'w') as f:
                        pass

                with open(os.path.join('../', 'data', 'already_scraped_links.txt'), 'r') as f:
                    self.scraped_links = [line.rstrip() for line in f.readlines()]

    def _process(self, **kwargs) -> None:
        """this method implements looping logic and expects `_scrape` method to return """
        i = 1
        chrome_options = Options()
        if kwargs.get('inference', False):
            chrome_options.add_argument("--headless")
        with webdriver.Chrome(ChromeDriverManager().install(),
                              chrome_options=chrome_options) as driver:  # this assumes that chrome is installed and was at least once launched
            driver.implicitly_wait(self.delay * 5)
            if self.prepared_links:  # used for scrapers based on specific urls
                for link in tqdm(self.prepared_links, desc='Scraping links...'):
                    if link not in self.scraped_links and link != '':
                        data = self.scrape(driver, link)
                        if data.get('status', None) is not None and data.get('status', None) == 'expired':
                            self.new_scraped_links.append(link)
                            self.scraped_links.append(link)
                            i += 1
                        elif data:
                            self.data.append(data)
                            self.new_scraped_links.append(link)
                            self.scraped_links.append(link)
                            i += 1
                    if i % 10 == 0:  # append to csv every 10 new rows
                        self._export_tabular_data(**kwargs)
                        self._update_scrapped_links()
                        self.data = []  # flush list
                        self.new_scraped_links = []

                self._export_tabular_data(**kwargs)
                self._update_scrapped_links(**kwargs)
            else:  # used for other scrappers which implement logic only in `scrape` method and return list of data
                self.data = self.scrape(driver, '')
                self._export_tabular_data(**kwargs)
    
    def run(self, **kwargs) -> None:
        """Main method used to connect whole logic"""
        self._prepare(**kwargs)
        self._process(**kwargs)

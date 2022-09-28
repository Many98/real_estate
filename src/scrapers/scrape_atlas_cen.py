import pandas as pd
import csv
import os
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from scrapers.BaseScraper import BaseScraper


class AtlasCenScraper(BaseScraper):
    def __init__(self, delay: float = 0.5) -> None:
        super().__init__(delay)

    def scrape(self, driver: WebDriver, url: str) -> dict:
        """Here comes custom implementation for atlascen
        """
        pass
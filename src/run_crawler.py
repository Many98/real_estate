from scrapers.crawl_sreality import KindOfCrawlerForSReality
from scrapers.crawl_breality import KindOfCrawlerForBezRealitky
from scrapers.scrape_sreality import SRealityScraper
from scrapers.scrape_breality import BezRealitkyScraper
import argparse

# TODO: async scrapers?

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description='Real estate scraper')
    #parser.add_argument('-c', '--config-name', help='Name of the config file', default='config.yaml')
    #arguments = parser.parse_args()

    # Here can be tested functionality of two scrapers
    #b = BezRealitkyScraper()
    #b.run(in_filename='prodej_links.txt', out_filename='prodej')
    s = SRealityScraper()
    s.run(in_filename='prodej_links.txt', out_filename='prodej')
    """
    sreality_crawler = KindOfCrawlerForSReality(out_filename='prodej_links.txt',
                                                url='https://www.sreality.cz/hledani/prodej/byty/praha?velikost=1%2Bkk,1%2B1&stav=pred-rekonstrukci,v-rekonstrukci')
    breality_crawler = KindOfCrawlerForBezRealitky(out_filename='prodej_links.txt',
                                                   url='https://www.bezrealitky.cz/vyhledat?offerType=PRODEJ&estateType=BYT&page=1&order=TIMEORDER_DESC&regionOsmId=R435514&osm_value=Hlavn%C3%AD+m%C4%9Bsto+Praha%2C+Praha%2C+%C4%8Cesko')

    sreality_crawler.crawl()
    breality_crawler.crawl()

    if sreality_crawler.reality_links:
        print(f'Found {len(sreality_crawler.reality_links)} new apartments at https://www.sreality.cz/!')

    if breality_crawler.reality_links:
        print(f'Found {len(breality_crawler.reality_links)} new apartments ar https://www.bezrealitky.cz/!')

    else:
        print('No new apartments found')
    """



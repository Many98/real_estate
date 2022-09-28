import os
from typing import List
from abc import ABC, abstractmethod


class BaseKindOfCrawler(ABC):
    """Abstract class for downloading url for apartment advertisement
        (therefore just kind of crawler and not proper crawler)"""
    def __init__(self,
                 out_filename: str,
                 url: str):

        self.out_filename = out_filename
        self.main_url = url
        self.reality_links = []
        self.filename = self.create_file()
        self.existing_links = self.get_existing_links()

    def create_file(self) -> str:
        '''creates the path to file where links to apts are saved if it doesn't exist, or outputs the path to existing file'''

        filename = os.path.join('../', 'data', self.out_filename)

        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                pass

        print(filename)

        return filename

    def get_existing_links(self) -> List:
        '''reads the file of existing links with apts and returns a list of existing apts'''

        with open(self.filename, 'r') as f:
            existing_links = [line.rstrip() for line in f.readlines()]

        print(f'Reading existing links from {self.filename}')

        return existing_links

    def append_to_txt(self, link: str = None) -> None:
        '''appends the apartment to the file with all apts'''

        # print(f'Appending link to {self.filename}')
        with open(self.filename, 'a') as f:
            f.write(link)
            f.write('\n')

    @abstractmethod
    def crawl(self, *args, **kwargs):
        """here implement custom logic"""
        pass

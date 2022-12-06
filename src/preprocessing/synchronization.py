import pandas as pd
import numpy as np
import os
import re


class Synchronizer(object):
    """
    Class handling unification/synchronization between sreality and bezrealitky data sources.
    It should also merge textual columns into one and
     ensure correct dtypes (string, floats, bool) within columns # TODO define exact dtypes for columns

    """

    def __init__(self, from_row: tuple[int, int]):
        """

        Parameters
        ----------
        from_row : tuple[int, int]
            Indices of row from which start. from_row[0] -> sreality
                                             from_row[1] -> breality TODO make this more robust
        """
        self.from_row = from_row
        self.final_df = pd.DataFrame()
        self.sreality_df = pd.DataFrame()
        self.breality_df = pd.DataFrame()

    def __call__(self, sreality_csv_path: str, breality_csv_path: str, inference: bool, *args, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        sreality_csv_path : str
            Path of sreality csv
        breality_csv_path : str
            Path of breality csv

        Returns
        -------

        """
        # dataframes to be synchronized
        try:
            self.sreality_df = pd.read_csv(sreality_csv_path).iloc[self.from_row[0]:, :]
        except:
            pass
        try:
            self.breality_df = pd.read_csv(breality_csv_path).iloc[self.from_row[1]:, :]
        except:
            pass

        try:
            if not self.sreality_df.empty or not self.breality_df.empty:
                if not self.sreality_df.empty:
                    self.extract_sreality_data()
                if not self.breality_df.empty:
                    self.extract_breality_data()
                    self.set_breality_dtypes()
                self.unify()
                self.merge_text()
                if not inference:
                    self.remove()
                    if self.integrity_check():
                        self.final_df.to_csv(os.path.join('..', 'data/tmp_synchronized.csv'), mode='w', index=False)
                else:
                    self.final_df.to_csv(os.path.join('..', 'data/tmp_synchronized.csv'), mode='w', index=False)
        except Exception as e:
            if not inference:
                print(e)

        return self.final_df

    def integrity_check(self) -> bool:
        """
        auxiliary function to check integrity
        Returns
        -------

        """
        assert self.final_df.ownership.isin(np.array(['Osobní', 'Státní/obecní', 'Družstevní', np.nan],
                                                     dtype=object)).all(), \
            f'`ownership` contains unexpected value {self.final_df.ownership.unique()}'
        assert self.final_df.price.min() > 1000 or self.final_df.price.min() is np.nan, '`price` must be positive'
        assert self.final_df.usable_area.min() > 0 or self.final_df.usable_area.min() is np.nan, '`usable_area` must be positive'
        assert -5 < self.final_df.floor.min() and self.final_df.floor.max() < 100 or self.final_df.floor.min() is np.nan,\
            '`floor` must be >-5'
        assert self.final_df.energy_effeciency.isin(np.array([np.nan, 'G', 'E', 'B', 'D', 'C',
                                                              'A', 'F'], dtype=object)).all(), \
            f'`energy_effeciency` contains unexpected value {self.final_df.energy_effeciency.unique()}'
        assert 13 < self.final_df.long.min() and self.final_df.long.max() < 16, '`long` must be within [14, 16]'
        assert 49 < self.final_df.lat.min() and self.final_df.lat.max() < 51, '`lat` must be within [49, 51]'

        assert self.final_df.equipment.isin(np.array(['ne', np.nan, 'Částečně', 'ano'], dtype=object)).all(), \
            f'`equipment` contains unexpected value {self.final_df.equipment.unique()}'
        assert self.final_df.state.isin(np.array([np.nan, 'V rekonstrukci', 'Před rekonstrukcí', 'Po rekonstrukci',
                                                  'Novostavba', 'Velmi dobrý', 'Dobrý', 'Ve výstavbě', 'Projekt',
                                                  'Špatný', ], dtype=object)).all(), \
            f'`state` contains unexpected value {self.final_df.state.unique()}'
        assert self.final_df.construction_type.isin(
            np.array([np.nan, 'Cihlová', 'Smíšená', 'Panelová', 'Skeletová', 'Kamenná',
                      'Montovaná', 'Nízkoenergetická', 'Drevostavba'], dtype=object)).all(), \
            f'`construction_type` contains unexpected value {self.final_df.construction_type.unique()}'
        assert self.final_df.disposition.isin(
            np.array([np.nan, '1+kk', '1+1', '3+1', '3+kk', '2+kk', '4+1', '2+1', '5+kk', '4+kk',
                      'atypické', '6 pokojů a více', '5+1', '6+kk'], dtype=object)).all(), \
            f'`disposition` contains unexpected value {self.final_df.disposition.unique()}'
        assert self.final_df.additional_disposition.isin(np.array([np.nan, 'Podkrovní', 'Loft', 'Mezonet'],
                                                                  dtype=object)).all(), \
            f'`additional_disposition` contains unexpected value {self.final_df.additional_disposition.unique()}'

        # TODO not useful
        #assert 0 <= self.final_df.year_reconstruction.min() or self.final_df.year_reconstruction.min() is np.nan, \
        #    '`year_reconstruction` must be within positive'

        # TODO this columns are not very useful #['gas', 'electricity', 'waste', 'heating', 'telecomunication']
        bool_cols = [col for col in self.final_df if 'has' in col and 'hash' not in col]
        for col in bool_cols:
            assert self.final_df[col].isin(np.array([np.nan, True, False])).all(), \
                f'`{col}` contains unexpected value {self.final_df[col].unique()}'

        dist_cols = [col for col in self.final_df if 'dist' in col]
        for col in dist_cols:
            assert self.final_df[col].min() >= 0 or self.final_df[col].min() is np.nan, f'`{col}` must be  positive'

        return True

    def _clean_description(self):
        """
        auxiliary method to clean a bit description column
        by removing price from it
        Returns
        -------

        """
        self.final_df['description'] = self.final_df['description'].apply(lambda x: re.sub(r'[0-9]{6,10}', '', x))
        for i in ['Zlevněno', 'Původní cena', 'price']:
            self.final_df['description'] = self.final_df['description'].apply(lambda x: re.sub(i, '', x))

    def _hash_it(self):
        """
        auxiliary method to add hashes based on every record values and based on description value
        Returns
        -------

        """
        self.final_df['hash'] = pd.util.hash_pandas_object(self.final_df, index=False).astype(str)
        self.final_df['desc_hash'] = pd.util.hash_pandas_object(self.final_df['description'], index=False).astype(str)

    def remove(self):
        """
        auxiliary method to remove some records with unexpected values e.g. price <=0; rental price instead
        of sellling price & also records with same hashes (remove duplicates)
        Returns
        -------

        """
        self.final_df.drop(self.final_df.loc[self.final_df['price'] <= 0].index, inplace=True)
        self.final_df.drop(self.final_df.loc[self.final_df['header'].str.contains('Pronájem')].index, inplace=True)

    def unify(self):
        """
        method to unify all levels within both dataframes and merge to one `final_df` dataframe
        Returns
        -------
        """
        self.final_df = pd.concat([self.sreality_df, self.breality_df], ignore_index=True)
        self.final_df = self.final_df.replace("nan", np.nan)

        self.final_df["ownership"] = self.final_df["ownership"].replace("OSOBNI", "Osobní")
        self.final_df["ownership"] = self.final_df["ownership"].replace("DRUZSTEVNI", "Družstevní")
        self.final_df["ownership"] = self.final_df["ownership"].replace("UNDEFINED", np.nan)
        self.final_df["ownership"] = self.final_df["ownership"].replace("OSTATNI", np.nan)

        self.final_df["equipment"] = self.final_df["equipment"].replace("VYBAVENY", "ano")
        self.final_df["equipment"] = self.final_df["equipment"].replace("NEVYBAVENY", "ne")
        self.final_df["equipment"] = self.final_df["equipment"].replace("VYBAVENY", "ano")
        self.final_df["equipment"] = self.final_df["equipment"].replace("CASTECNE", "Částečně")

        self.final_df["construction_type"] = self.final_df["construction_type"].replace("CIHLA", "Cihlová")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("OSTATNI", "Smíšená")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("PANEL", "Panelová")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("NIZKOENERGETICKY",
                                                                                        "Nízkoenergetická")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("DREVOSTAVBA",
                                                                                        "Drevostavba")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("Dřevěná",
                                                                                        "Drevostavba")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("UNDEFINED", np.nan)

        self.final_df["state"] = self.final_df["state"].replace("UNDEFINED", np.nan)
        self.final_df["state"] = self.final_df["state"].replace("VERY_GOOD", 'Velmi dobrý')
        self.final_df["state"] = self.final_df["state"].replace("GOOD", 'Dobrý')
        self.final_df["state"] = self.final_df["state"].replace("NEW", 'Novostavba')
        self.final_df["state"] = self.final_df["state"].replace("BAD", 'Špatný')
        self.final_df["state"] = self.final_df["state"].replace("CONSTRUCTION", 'V rekonstrukci')

        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_6_KK", "6+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_2_KK", "2+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_4_KK", "4+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_1_1", "1+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_3_KK", "3+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_1_KK", "1+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_2_1", "2+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_2_IZB", "2+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_4_1", "4+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_5_KK", "5+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_3_1", "3+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("OSTATNI", "atypické")
        self.final_df["disposition"] = self.final_df["disposition"].replace("GARSONIERA", "1+kk")
        self.final_df["disposition"] = self.final_df["disposition"].replace("DISP_5_1", "5+1")
        self.final_df["disposition"] = self.final_df["disposition"].replace("UNDEFINED", "atypické")

        self._hash_it()

    def merge_text(self):
        """
        method for merging all columns which should be processed by NLP
        * description (textual attribute => NLP)
        * note (from sreality data) => NLP
        * tags (from bezrealitky data) => NLP
        * place (from bezrealitky data) => NLP
        * transport
        * telecommunication
        Returns
        -------

        """
        self._clean_description()  # remove prices from description

        cols = ['note', 'tags', 'place', 'transport', 'telecomunication_txt', 'heating_txt', 'additional_disposition',
                'waste_txt', 'electricity_txt']
        for col in cols:
            self.final_df[col].fillna('', inplace=True)
            self.final_df['description'] += ' ' + self.final_df[col]
            self.final_df[col] = self.final_df[col].replace('', np.nan)

    def set_breality_dtypes(self):
        """
        method to check and ensure correct dtypes on relevant columns on both dataframes
        Returns
        -------

        """
        self.breality_df = self.breality_df.replace("nan", np.nan)

        self.breality_df["price"] = self.breality_df["price"].astype("float64")
        self.breality_df["note"] = self.breality_df["note"].astype("str")
        self.breality_df["usable_area"] = self.breality_df["usable_area"].astype("float64")
        self.breality_df["gas"] = self.breality_df["gas"].astype("str")
        self.breality_df["waste"] = self.breality_df["waste"].astype("str")
        self.breality_df["electricity"] = self.breality_df["electricity"].astype("str")
        self.breality_df["transport"] = self.breality_df["transport"].astype("str")
        self.breality_df["telecomunication"] = self.breality_df["telecomunication"].astype("str")
        self.breality_df["tags"] = self.breality_df["tags"].astype("str")
        self.breality_df["waste_txt"] = self.breality_df["waste_txt"].astype("str")
        self.breality_df["electricity_txt"] = self.breality_df["electricity_txt"].astype("str")
        self.breality_df["telecomunication_txt"] = self.breality_df["telecomunication_txt"].astype("str")
        self.breality_df["additional_disposition"] = self.breality_df["additional_disposition"].astype("str")

    def extract_breality_data(self) -> None:
        """
        method to hold consistency and for creation os some additional columns
        Returns
        -------

        """

        self.breality_df['waste_txt'] = self.breality_df['waste']
        self.breality_df['electricity_txt'] = self.breality_df['electricity']
        self.breality_df['heating_txt'] = self.breality_df['heating']
        self.breality_df['telecomunication_txt'] = self.breality_df['telecomunication']

        self.breality_df['additional_disposition'] = np.array([np.nan] * self.breality_df.shape[0])

        self.breality_df["has_cellar"] = self.breality_df["has_cellar"].apply(
            lambda x: bool(x) if pd.notnull(x) else False)
        self.breality_df["has_loggia"] = self.breality_df["has_loggia"].apply(
            lambda x: bool(x) if pd.notnull(x) and x else False)
        self.breality_df["has_balcony"] = self.breality_df["has_balcony"].apply(
            lambda x: bool(x) if pd.notnull(x) else False)

        self.breality_df["has_garden"] = self.breality_df["has_garden"].apply(
            lambda x: bool(x) if pd.notnull(x) else False)

        self.breality_df["has_parking"] = self.breality_df["has_parking"].apply(
            lambda x: bool(x) if pd.notnull(x) else False)

        self.breality_df['heating'] = self.breality_df['heating'].apply(
            lambda x: bool(x) if pd.notnull(x) else np.nan)

    def extract_sreality_data(self) -> None:
        """
        auxiliary method to extract data from sreality because most of them were in string format
        Returns
        -------

        """
        try:
            # header
            self.sreality_df['header'] = self.sreality_df['header'].apply(
                lambda x: x.replace("\xa0", " ") if pd.notnull(x) else np.nan).astype('str')

            # price
            self.sreality_df['price'] = self.sreality_df['price'].apply(
                lambda x: re.sub(r'[^0-9]', '', x.split('K')[0]) if pd.notnull(x) else np.nan)
            self.sreality_df['price'] = self.sreality_df['price'].apply(
                lambda x: x if x != '' else np.nan).astype('float')

            # usable area
            self.sreality_df['usable_area'] = self.sreality_df['usable_area'].apply(
                lambda x: re.sub(r'[^0-9]', '', x.replace('m2', '')) if pd.notnull(x) else np.nan)
            self.sreality_df['usable_area'] = self.sreality_df['usable_area'].apply(
                lambda x: x if x != '' else np.nan).astype('float')

            # floor
            self.sreality_df['floor'] = self.sreality_df['floor'].apply(
                lambda x: x.replace('přízemí', '0.') if pd.notnull(x) else np.nan)
            self.sreality_df['floor'] = self.sreality_df['floor'].apply(
                lambda x: x.split('.')[0] if pd.notnull(x) else np.nan).astype(float)

            # energy efficiency
            self.sreality_df['energy_effeciency'] = self.sreality_df['energy_effeciency'].apply(
                lambda x: str(x)[6] if pd.notnull(x) else np.nan)

            # long
            self.sreality_df['long'] = self.sreality_df['long'].apply(
                lambda x: x[2:] if pd.notnull(x) else np.nan).astype('float')

            # lat
            self.sreality_df['lat'] = self.sreality_df['lat'].apply(
                lambda x: x[2:] if pd.notnull(x) else np.nan).astype('float')

            #  <>_dist cols
            dist_cols = [i for i in self.sreality_df.columns if 'dist' in i]
            for col in dist_cols:
                self.sreality_df[col] = self.sreality_df[col].apply(
                    lambda x: re.sub(r'[^0-9]', '', str(x).split('(')[-1]) if pd.notnull(x) else np.nan)
                self.sreality_df[col] = self.sreality_df[col].apply(
                    lambda x: x if pd.notnull(x) and x != '' else np.nan).astype('float')

            # gas
            self.sreality_df['gas'] = self.sreality_df['gas'].apply(
                lambda x: bool(x) if pd.notnull(x) else np.nan)

            # waste
            self.sreality_df['waste_txt'] = self.sreality_df['waste']
            self.sreality_df['waste'] = self.sreality_df['waste'].apply(
                lambda x: bool(x) if pd.notnull(x) else np.nan)

            # electricity
            self.sreality_df['electricity_txt'] = self.sreality_df['electricity']
            self.sreality_df['electricity'] = self.sreality_df['electricity'].apply(
                lambda x: bool(x) if pd.notnull(x) else np.nan)

            # heating
            self.sreality_df['heating_txt'] = self.sreality_df['heating']
            self.sreality_df['heating'] = self.sreality_df['heating'].apply(
                lambda x: bool(x) if pd.notnull(x) else np.nan)

            # telecomunication
            self.sreality_df['telecomunication_txt'] = self.sreality_df['telecomunication']
            self.sreality_df['telecomunication'] = self.sreality_df['telecomunication'].apply(
                lambda x: bool(x) if pd.notnull(x) else np.nan)

            # disposition
            self.sreality_df['additional_disposition'] = self.sreality_df['header'].apply(
                lambda x: x.split('(')[-1].split(')')[0].replace(x.split('(')[0].split(')')[0],
                                                                 '') if pd.notnull(x) else np.nan)
            self.sreality_df['additional_disposition'] = self.sreality_df['additional_disposition'].replace('', np.nan)
            disp = self.sreality_df['header'].str.extract(
                r'(1\+kk)|(1\+1)|(3\+1)|(3\+kk)|(2\+kk)|(4\+1)|(2\+1)|(5\+kk)|(4\+kk)|(atypické)|(6 pokojů a více)|(5\+1)')
            disp.fillna('', inplace=True)
            self.sreality_df['disposition'] = disp.sum(axis=1).replace('', np.nan).astype(str)

            # equipment
            # ok because its string features

            # state
            # ok because its string features

            # construction type
            # ok because its string features

            # year reconstruction
            self.sreality_df['year_reconstruction'] = self.sreality_df['year_reconstruction'].astype(float)

            #  `has_<>` columns
            has_cols = [i for i in self.sreality_df.columns if 'has' in i and 'hash' not in i]
            for col in has_cols:
                self.sreality_df[col] = self.sreality_df[col].apply(
                    lambda x: str(x).replace('Topení:', '') if pd.notnull(x) else np.nan)
                self.sreality_df[col] = self.sreality_df[col].apply(
                    lambda x: True if pd.notnull(x) and x not in ('', 'ne') else False)
                if col == "has_garden":
                    self.sreality_df[col] = False
        except:
            return


if __name__ == '__main__':
    synchronizer = Synchronizer(from_row=(0, 0))
    synchronizer(sreality_csv_path=os.path.join('..', '..', 'data/prodej_sreality_scraped.csv'),
                 breality_csv_path=os.path.join('..', '..', 'data/prodej_breality_scraped.csv'))

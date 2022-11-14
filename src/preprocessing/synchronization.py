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
        self.final_df = None

    def __call__(self, sreality_csv_path: str, breality_csv_path: str, *args, **kwargs) -> pd.DataFrame:
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
        try:
            self.sreality_df = pd.read_csv(sreality_csv_path).iloc[self.from_row[0]:, :]
            self.breality_df = pd.read_csv(breality_csv_path).iloc[self.from_row[1]:,
                               :]  # dataframes to be synchronized
        except Exception as e:
            print(e)

        self.extract_sreality_data()
        self.extract_breality_data()
        self.check_dtypes()  # checks dtypes on both dataframes
        self.unify()
        self.merge_text()

        self.final_df.to_csv(os.path.join('..', '..', 'data/tmp_synchronized.csv'), mode='w', index=False)

        return self.final_df

    def unify(self):
        """
        method to unify all levels within both dataframes and merge to one `final_df` dataframe
        Returns
        -------
        """
        # TODO HERE PERFORM UNIFICATION ON DATAFRAMES `sreality_df` and `breality_df`
        #  i.e. there cannot be redundancy i.e. instead of result
        #  of `self.final_df['ownership'].unique()` `array(['Osobní', 'Státní/obecní', 'Družstevní', nan, 'OSOBNI',
        #        'UNDEFINED', 'DRUZSTEVNI'])`
        #  we want to have just `array(['Osobní', 'Státní/obecní', 'Družstevní', nan])`
        #  and similar for other columns

        # here add your code
        self.final_df = pd.concat([self.sreality_df, self.breality_df])
        self.final_df = self.final_df.replace("nan", np.nan)
        self.final_df["ownership"] = self.final_df["ownership"].replace("OSOBNI", "Osobní")
        self.final_df["ownership"] = self.final_df["ownership"].replace("DRUZSTEVNI", "Družstevní")
        self.final_df["ownership"] = self.final_df["ownership"].replace("UNDEFINED", np.nan)
        self.final_df["equipment"] = self.final_df["equipment"].replace("VYBAVENY", "ano")
        self.final_df["equipment"] = self.final_df["equipment"].replace("NEVYBAVENY", "ne")
        self.final_df["equipment"] = self.final_df["equipment"].replace("VYBAVENY", "ano")
        self.final_df["equipment"] = self.final_df["equipment"].replace("CASTECNE", "Částečně")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("CIHLA", "Cihlová")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("OSTATNI", "Smíšená")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("PANEL", "Panelová")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("NIZKOENERGETICKY", "Nízkoenergetická")
        self.final_df["construction_type"] = self.final_df["construction_type"].replace("UNDEFINED", np.nan)
        self.final_df["condition"] = self.final_df["condition"].replace("UNDEFINED", np.nan)
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

        pass

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
        cols = ['note', 'tags', 'place', 'transport', 'telecomunication_txt', 'heating_txt', 'additional_disposition',
                'waste_txt', 'electricity_txt']
        for col in cols:
            self.final_df['description'] += ' ' + self.final_df[col]

    def check_dtypes(self):
        """
        method to check and ensure correct dtypes on relevant columns on both dataframes
        Returns
        -------

        """
        # TODO HERE check dtypes on both dataframes or on final its up to you
        #  e.g. with assert statemnets
        self.breality_df["price"] = self.breality_df["price"].astype("float64")
        self.sreality_df["note"] = self.sreality_df["note"].astype("str")
        self.breality_df["note"] = self.breality_df["note"].astype("str")
        self.breality_df["usable_area"] = self.breality_df["usable_area"].astype("float64")
        self.breality_df["gas"] = self.breality_df["gas"].astype("str")
        self.breality_df["waste"] = self.breality_df["waste"].astype("str")
        self.breality_df["electricity"] = self.breality_df["electricity"].astype("str")
        self.breality_df["transport"] = self.breality_df["transport"].astype("str")
        self.breality_df["telecomunication"] = self.breality_df["telecomunication"].astype("str")
        self.sreality_df["tags"] = self.sreality_df["tags"].astype("str")
        self.breality_df["waste_txt"] = self.breality_df["waste_txt"].astype("str")
        self.breality_df["electricity_txt"] = self.breality_df["electricity_txt"].astype("str")
        self.breality_df["telecomunication_txt"] = self.breality_df["telecomunication_txt"].astype("str")
        self.breality_df["additional_disposition"] = self.breality_df["additional_disposition"].astype("str")



        pass

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

        self.breality_df["has_cellar"] = self.sreality_df["has_cellar"].apply(
            lambda x: True if x == True else False)
        self.breality_df["has_loggia"] = self.sreality_df["has_loggia"].apply(
            lambda x: True if x == True else False)

        self.breality_df["has_garden"] = self.breality_df["has_garden"].apply(
            lambda x: True if x is not np.nan else np.nan)

        self.breality_df["has_parking"] = self.breality_df["has_parking"].apply(
            lambda x: True if x == True else False)
        self.breality_df["MHD"] = self.breality_df["MHD"].apply(
            lambda x: True if x == True else False)
        self.breality_df["post_office"] = self.breality_df["post_office"].apply(
            lambda x: True if x == True else False)

        self.breality_df["bank"] = self.breality_df["bank"].apply(
            lambda x: True if x == True else False)
        self.breality_df["school"] = self.breality_df["school"].apply(
            lambda x: True if x == True else False)
        self.breality_df["kindergarten"] = self.breality_df["kindergarten"].apply(
            lambda x: True if x == True else False)
        self.breality_df["supermarket_grocery"] = self.breality_df["supermarket_grocery"].apply(
            lambda x: True if x == True else False)
        self.breality_df["restaurant_pub"] = self.breality_df["restaurant_pub"].apply(
            lambda x: True if x == True else False)
        self.breality_df["playground"] = self.breality_df["playground"].apply(
            lambda x: True if x == True else False)
        self.breality_df["sports_field"] = self.breality_df["sports_field"].apply(
            lambda x: True if x == True else False)
        self.breality_df["pharmacy"] = self.breality_df["pharmacy"].apply(
            lambda x: True if x == True else False)

        self.breality_df['heating'] = self.breality_df['heating'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

    def extract_sreality_data(self) -> None:
        """
        auxiliary method to extract data from sreality because most of them were in string format
        Returns
        -------

        """
        # header
        self.sreality_df['header'] = self.sreality_df['header'].apply(
            lambda x: x.replace("\xa0", " ") if x is not np.nan else np.nan).astype('str')

        # price
        self.sreality_df['price'] = self.sreality_df['price'].apply(
            lambda x: re.sub(r'[^0-9]', '', x.split('K')[0]) if x is not np.nan else np.nan)
        self.sreality_df['price'] = self.sreality_df['price'].apply(
            lambda x: x if x != '' else np.nan).astype('float')

        # usable area
        self.sreality_df['usable_area'] = self.sreality_df['usable_area'].apply(
            lambda x: re.sub(r'[^0-9]', '', x.replace('m2', '')) if x is not np.nan else np.nan)
        self.sreality_df['usable_area'] = self.sreality_df['usable_area'].apply(
            lambda x: x if x != '' else np.nan).astype('float')

        # floor
        self.sreality_df['floor'] = self.sreality_df['floor'].apply(
            lambda x: x.replace('přízemí', '0.') if x is not np.nan else np.nan)
        self.sreality_df['floor'] = self.sreality_df['floor'].apply(
            lambda x: x.split('.')[0] if x is not np.nan else np.nan).astype(float)

        # energy efficiency
        self.sreality_df['energy_effeciency'] = self.sreality_df['energy_effeciency'].apply(
            lambda x: str(x[6]) if x is not np.nan else np.nan)

        # long
        self.sreality_df['long'] = self.sreality_df['long'].apply(
            lambda x: x[2:] if x is not np.nan else np.nan).astype('float')

        # lat
        self.sreality_df['lat'] = self.sreality_df['lat'].apply(
            lambda x: x[2:] if x is not np.nan else np.nan).astype('float')

        #  <>_dist cols
        dist_cols = [i for i in self.sreality_df.columns if 'dist' in i]
        for col in dist_cols:
            self.sreality_df[col] = self.sreality_df[col].apply(
                lambda x: re.sub(r'[^0-9]', '', str(x).split('(')[-1]) if x is not np.nan else np.nan)
            self.sreality_df[col] = self.sreality_df[col].apply(
                lambda x: x if x is not np.nan and x != '' else np.nan).astype('float')

        # gas
        self.sreality_df['gas'] = self.sreality_df['gas'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

        # waste
        self.sreality_df['waste_txt'] = self.sreality_df['waste']
        self.sreality_df['waste'] = self.sreality_df['waste'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

        # electricity
        self.sreality_df['electricity_txt'] = self.sreality_df['electricity']
        self.sreality_df['electricity'] = self.sreality_df['electricity'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

        # heating
        self.sreality_df['heating_txt'] = self.sreality_df['heating']
        self.sreality_df['heating'] = self.sreality_df['heating'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

        # telecomunication
        self.sreality_df['telecomunication_txt'] = self.sreality_df['telecomunication']
        self.sreality_df['telecomunication'] = self.sreality_df['telecomunication'].apply(
            lambda x: bool(x) if x is not np.nan else np.nan)

        # disposition
        self.sreality_df['additional_disposition'] = self.sreality_df['header'].apply(
            lambda x: x.split('(')[-1].split(')')[0].replace(x.split('(')[0].split(')')[0],
                                                             '') if x is not np.nan else np.nan)

        self.sreality_df['disposition'] = self.sreality_df['header'].apply(
            lambda x: x.split()[2] if x is not np.nan else np.nan)

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
                lambda x: str(x).replace('Topení:', '') if x is not np.nan else np.nan)
            self.sreality_df[col] = self.sreality_df[col].apply(
                lambda x: True if x is not np.nan and x not in ('', 'ne') else False)
            if col == "has_garden":
                self.sreality_df[col] = False


synchronizer = Synchronizer(tuple([0, 0]))
synchronizer(sreality_csv_path=os.path.join('..', '..', 'data/prodej_sreality_scraped.csv'),
             breality_csv_path=os.path.join('..', '..', 'data/prodej_breality_scraped.csv'))

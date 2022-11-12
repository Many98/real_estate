import pandas as pd
import numpy as np
import os


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
            self.breality_df = pd.read_csv(breality_csv_path).iloc[self.from_row[1]:, :]  # dataframes to be synchronized
        except Exception as e:
            print(e)

        self.dictionary_sreality = self.sreality_df.to_dict()
        self.dictionary_breality = self.breality_df.to_dict()
        self.dictionary_sreality = self.unify_categoric_variables()
        self.dictionary_sreality, self.dictionary_breality = self.merge_text()
        self.check_dtypes()  # checks dtypes on both dataframes
        self.unify()
        #self.merge_text()

        self.final_df.to_csv('C:/Users/adams/OneDrive/Dokumenty/GitHub/real_estate/data/tmp_synchronized.csv', mode='w', index=False)

        return self.final_df

    def unify(self):
        """
        method to unify all levels within both dataframes and merge to one `final_df` dataframe
        Returns
        -------
        """
        self.final_df = pd.concat([self.sreality_data, self.breality_data])

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
        for i in range(len(self.dictionary_sreality["description"])):
            if self.dictionary_sreality["note"][i] is not None:
                self.dictionary_sreality["description"][i] = self.dictionary_sreality["description"][i] + str(self.dictionary_sreality["note"][i])
            if self.dictionary_sreality["transport"][i] is not None:
                self.dictionary_sreality["description"][i] = self.dictionary_sreality["description"][i] + str(self.dictionary_sreality["transport"][i])
            if self.dictionary_sreality["telecomunication"][i] is not None:
                self.dictionary_sreality["description"][i] = self.dictionary_sreality["description"][i] + str(self.dictionary_sreality["telecomunication"][i])
        for i in range(len(self.dictionary_breality["description"])):
            if self.dictionary_breality["note"][i] is not None:
                self.dictionary_breality["description"][i] = self.dictionary_breality["description"][i] + str(self.dictionary_breality["note"][i])
            if self.dictionary_breality["tags"][i] is not None:
                self.dictionary_breality["description"][i] = self.dictionary_breality["description"][i] + str(self.dictionary_breality["tags"][i])
            if self.dictionary_breality["place"][i] is not None:
                self.dictionary_breality["description"][i] = self.dictionary_breality["description"][i] + \
                                                         str(self.dictionary_breality["place"][i])
            if self.dictionary_breality["transport"][i] is not None:
                self.dictionary_breality["description"][i] = self.dictionary_breality["description"][i] + str(self.dictionary_breality["transport"][i])
            if self.dictionary_breality["telecomunication"][i] is not None:
                self.dictionary_breality["description"][i] = self.dictionary_breality["description"][i] + str(self.dictionary_breality["telecomunication"][i])
        return self.dictionary_sreality, self.dictionary_breality

    def check_dtypes(self):
        """
        method to check and ensure correct dtypes on relevant columns on both dataframes
        Returns
        -------

        """
        self.sreality_data = pd.DataFrame.from_dict(self.dictionary_sreality)
        self.breality_data = pd.DataFrame.from_dict(self.dictionary_breality)

        # sreality types
        self.sreality_data["price"] = self.sreality_data["price"].fillna(0).astype("int64")
        self.sreality_data["usable_area"] = self.sreality_data["usable_area"].fillna(0).astype("int64")
        # self.sreality_data["floor"] = self.sreality_data["floor"].astype("int64")



        pass

    def unify_categoric_variables(self):


        # header
        for i in range(len(self.dictionary_sreality["header"])):
            self.dictionary_sreality["header"][i] = self.dictionary_sreality["header"][i].replace("\xa0", " ")


        # price
        for i in range(len(self.dictionary_sreality["price"])):
            try:
                cislo = int(self.dictionary_sreality["price"][i][0])
                self.dictionary_sreality["price"][i] = self.dictionary_sreality["price"][i].replace(" ", "")
                index = self.dictionary_sreality["price"][i].find("K")
                self.dictionary_sreality["price"][i] = int(self.dictionary_sreality["price"][i][:index])
            except:
                self.dictionary_sreality["price"][i] = None


        # usable area
        cellar_area = {}
        for i in range(len(self.dictionary_sreality["usable_area"])):
            cellar_area[i] = None
            try:
                cislo = int(self.dictionary_sreality["usable_area"][i][0])
                self.dictionary_sreality["usable_area"][i] = self.dictionary_sreality["usable_area"][i].replace(" ", "")
                index = self.dictionary_sreality["usable_area"][i].find("m")
                self.dictionary_sreality["usable_area"][i] = int(self.dictionary_sreality["usable_area"][i][:index])
            except:
                self.dictionary_sreality["usable_area"][i] = None
        self.dictionary_sreality["cellar_area"] = cellar_area

        # floor
        for i in range(len(self.dictionary_sreality["floor"])):
            try:
                index = self.dictionary_sreality["floor"][i].find(" ")
                if "přízemí" in self.dictionary_sreality["floor"][i]:
                    self.dictionary_sreality["floor"][i] = 0
                elif self.dictionary_sreality["floor"][i][index-1] == ".":
                    self.dictionary_sreality["floor"][i] = int(self.dictionary_sreality["floor"][i][:(index - 1)])
            except:
                pass



        # energy efficiency
        for i in range(len(self.dictionary_sreality["energy_effeciency"])):
            try:
                self.dictionary_sreality["energy_effeciency"][i] = self.dictionary_sreality["energy_effeciency"][i][6]
            except:
                pass


        # long
        for i in range(len(self.dictionary_sreality["long"])):
            if type(self.dictionary_sreality["long"]) is not None:
                self.dictionary_sreality["long"][i] = float(self.dictionary_sreality["long"][i][2:])


        # lat
        for i in range(len(self.dictionary_sreality["lat"])):
            if type(self.dictionary_sreality["lat"]) is not None:
                self.dictionary_sreality["lat"][i] = float(self.dictionary_sreality["lat"][i][2:])


        # bus station dist
        for i in range(len(self.dictionary_sreality["bus_station_dist"])):
            try:
                index = self.dictionary_sreality["bus_station_dist"][i].find("(")
                self.dictionary_sreality["bus_station_dist"][i] = self.dictionary_sreality["bus_station_dist"][i][(index+1):]
                index = self.dictionary_sreality["bus_station_dist"][i].find(" ")
                self.dictionary_sreality["bus_station_dist"][i] = float(self.dictionary_sreality["bus_station_dist"][i][:index])
            except:
                pass


        # train station dist
        for i in range(len(self.dictionary_sreality["train_station_dist"])):
            try:
                index = self.dictionary_sreality["train_station_dist"][i].find("(")
                self.dictionary_sreality["train_station_dist"][i] = self.dictionary_sreality["train_station_dist"][i][(index+1):]
                index = self.dictionary_sreality["train_station_dist"][i].find(" ")
                self.dictionary_sreality["train_station_dist"][i] = float(self.dictionary_sreality["train_station_dist"][i][:index])
            except:
                pass

        # subway station dist
        for i in range(len(self.dictionary_sreality["subway_station_dist"])):
            try:
                index = self.dictionary_sreality["subway_station_dist"][i].find("(")
                self.dictionary_sreality["subway_station_dist"][i] = self.dictionary_sreality["subway_station_dist"][i][(index+1):]
                index = self.dictionary_sreality["subway_station_dist"][i].find(" ")
                self.dictionary_sreality["subway_station_dist"][i] = float(self.dictionary_sreality["subway_station_dist"][i][:index])
            except:
                pass

        # tram station dist
        for i in range(len(self.dictionary_sreality["tram_station_dist"])):
            try:
                index = self.dictionary_sreality["tram_station_dist"][i].find("(")
                self.dictionary_sreality["tram_station_dist"][i] = self.dictionary_sreality["tram_station_dist"][i][(index+1):]
                index = self.dictionary_sreality["tram_station_dist"][i].find(" ")
                self.dictionary_sreality["tram_station_dist"][i] = float(self.dictionary_sreality["tram_station_dist"][i][:index])
            except:
                pass

        # post office dist
        for i in range(len(self.dictionary_sreality["post_office_dist"])):
            try:
                index = self.dictionary_sreality["post_office_dist"][i].find("(")
                self.dictionary_sreality["post_office_dist"][i] = self.dictionary_sreality["post_office_dist"][i][(index+1):]
                index = self.dictionary_sreality["post_office_dist"][i].find(" ")
                self.dictionary_sreality["post_office_dist"][i] = float(self.dictionary_sreality["post_office_dist"][i][:index])
            except:
                pass

        # atm dist
        for i in range(len(self.dictionary_sreality["atm_dist"])):
            try:
                index = self.dictionary_sreality["atm_dist"][i].find("(")
                self.dictionary_sreality["atm_dist"][i] = self.dictionary_sreality["atm_dist"][i][(index+1):]
                index = self.dictionary_sreality["atm_dist"][i].find(" ")
                self.dictionary_sreality["atm_dist"][i] = float(self.dictionary_sreality["atm_dist"][i][:index])
            except:
                pass

        # doctor dist
        for i in range(len(self.dictionary_sreality["doctor_dist"])):
            try:
                index = self.dictionary_sreality["doctor_dist"][i].find("(")
                self.dictionary_sreality["doctor_dist"][i] = self.dictionary_sreality["doctor_dist"][i][(index+1):]
                index = self.dictionary_sreality["doctor_dist"][i].find(" ")
                self.dictionary_sreality["doctor_dist"][i] = float(self.dictionary_sreality["doctor_dist"][i][:index])
            except:
                pass

        # vet dist
        for i in range(len(self.dictionary_sreality["vet_dist"])):
            try:
                index = self.dictionary_sreality["vet_dist"][i].find("(")
                self.dictionary_sreality["vet_dist"][i] = self.dictionary_sreality["vet_dist"][i][(index+1):]
                index = self.dictionary_sreality["vet_dist"][i].find(" ")
                self.dictionary_sreality["vet_dist"][i] = float(self.dictionary_sreality["vet_dist"][i][:index])
            except:
                pass

        # primary school dist
        for i in range(len(self.dictionary_sreality["primary_school_dist"])):
            try:
                index = self.dictionary_sreality["primary_school_dist"][i].find("(")
                self.dictionary_sreality["primary_school_dist"][i] = self.dictionary_sreality["primary_school_dist"][i][(index+1):]
                index = self.dictionary_sreality["primary_school_dist"][i].find(" ")
                self.dictionary_sreality["primary_school_dist"][i] = float(self.dictionary_sreality["primary_school_dist"][i][:index])
            except:
                pass

        # kindergarten dist
        for i in range(len(self.dictionary_sreality["kindergarten_dist"])):
            try:
                index = self.dictionary_sreality["kindergarten_dist"][i].find("(")
                self.dictionary_sreality["kindergarten_dist"][i] = self.dictionary_sreality["kindergarten_dist"][i][(index+1):]
                index = self.dictionary_sreality["kindergarten_dist"][i].find(" ")
                self.dictionary_sreality["kindergarten_dist"][i] = float(self.dictionary_sreality["kindergarten_dist"][i][:index])
            except:
                pass

        # supermarket grocery dist
        for i in range(len(self.dictionary_sreality["supermarket_grocery_dist"])):
            try:
                index = self.dictionary_sreality["supermarket_grocery_dist"][i].find("(")
                self.dictionary_sreality["supermarket_grocery_dist"][i] = self.dictionary_sreality["supermarket_grocery_dist"][i][(index+1):]
                index = self.dictionary_sreality["supermarket_grocery_dist"][i].find(" ")
                self.dictionary_sreality["supermarket_grocery_dist"][i] = float(self.dictionary_sreality["supermarket_grocery_dist"][i][:index])
            except:
                pass

        # restaurant, pub dist
        for i in range(len(self.dictionary_sreality["restaurant_pub_dist"])):
            try:
                index = self.dictionary_sreality["restaurant_pub_dist"][i].find("(")
                self.dictionary_sreality["restaurant_pub_dist"][i] = self.dictionary_sreality["restaurant_pub_dist"][i][(index+1):]
                index = self.dictionary_sreality["restaurant_pub_dist"][i].find(" ")
                self.dictionary_sreality["restaurant_pub_dist"][i] = float(self.dictionary_sreality["restaurant_pub_dist"][i][:index])
            except:
                pass

        # playground dist
        for i in range(len(self.dictionary_sreality["playground_dist"])):
            try:
                index = self.dictionary_sreality["playground_dist"][i].find("(")
                self.dictionary_sreality["playground_dist"][i] = self.dictionary_sreality["playground_dist"][i][(index+1):]
                index = self.dictionary_sreality["playground_dist"][i].find(" ")
                self.dictionary_sreality["playground_dist"][i] = float(self.dictionary_sreality["playground_dist"][i][:index])
            except:
                pass

        # sports field dist
        for i in range(len(self.dictionary_sreality["sports_field_dist"])):
            try:
                index = self.dictionary_sreality["sports_field_dist"][i].find("(")
                self.dictionary_sreality["sports_field_dist"][i] = self.dictionary_sreality["sports_field_dist"][i][(index+1):]
                index = self.dictionary_sreality["sports_field_dist"][i].find(" ")
                self.dictionary_sreality["sports_field_dist"][i] = float(self.dictionary_sreality["sports_field_dist"][i][:index])
            except:
                pass

        # theatre cinema dist
        for i in range(len(self.dictionary_sreality["theatre_cinema_dist"])):
            try:
                index = self.dictionary_sreality["theatre_cinema_dist"][i].find("(")
                self.dictionary_sreality["theatre_cinema_dist"][i] = \
                self.dictionary_sreality["theatre_cinema_dist"][i][(index + 1):]
                if self.dictionary_sreality["theatre_cinema_dist"][i].find("(") >= 0:
                    index = self.dictionary_sreality["theatre_cinema_dist"][i].find("(")
                    self.dictionary_sreality["theatre_cinema_dist"][i] = \
                    self.dictionary_sreality["theatre_cinema_dist"][i][(index + 1):]
                index = self.dictionary_sreality["theatre_cinema_dist"][i].find(" ")
                self.dictionary_sreality["theatre_cinema_dist"][i] = float(self.dictionary_sreality["theatre_cinema_dist"][i][:index])
            except:
                pass

        # pharmacy dist
        for i in range(len(self.dictionary_sreality["pharmacy_dist"])):
            try:
                index = self.dictionary_sreality["pharmacy_dist"][i].find("(")
                self.dictionary_sreality["pharmacy_dist"][i] = self.dictionary_sreality["pharmacy_dist"][i][(index+1):]
                index = self.dictionary_sreality["pharmacy_dist"][i].find(" ")
                self.dictionary_sreality["pharmacy_dist"][i] = float(self.dictionary_sreality["pharmacy_dist"][i][:index])
            except:
                pass

        # gas
        for i in range(len(self.dictionary_sreality["gas"])):
            if str(self.dictionary_sreality["gas"][i]) == 'nan':
                self.dictionary_sreality["gas"][i] = None
            self.dictionary_sreality["gas"][i] = bool(self.dictionary_sreality["gas"][i])
            self.dictionary_sreality["gas"][i] = float(self.dictionary_sreality["gas"][i])

        # waste
        for i in range(len(self.dictionary_sreality["waste"])):
            if str(self.dictionary_sreality["waste"][i]) == "nan":
                self.dictionary_sreality["waste"][i] = None
            self.dictionary_sreality["waste"][i] = bool(self.dictionary_sreality["waste"][i])
            self.dictionary_sreality["waste"][i] = float(self.dictionary_sreality["waste"][i])

        # year reconstruction
        for i in range(len(self.dictionary_sreality["year_reconstruction"])):
            try:
                self.dictionary_sreality["year_reconstruction"][i] = int(self.dictionary_sreality["year_reconstruction"][i])
            except:
                pass

        # has garage
        for i in range(len(self.dictionary_sreality["has_garage"])):
            if self.dictionary_sreality["has_garage"][i] == "Topení:" or str(self.dictionary_sreality["has_garage"][i]) == 'nan':
                self.dictionary_sreality["has_garage"][i] = None
            self.dictionary_sreality["has_garage"][i] = bool(self.dictionary_sreality["has_garage"][i])
            self.dictionary_sreality["has_garage"][i] = float(self.dictionary_sreality["has_garage"][i])

        # has lift
        print(set(self.dictionary_sreality["has_lift"].values()))
        for i in range(len(self.dictionary_sreality["has_lift"])):
            if str(self.dictionary_sreality["has_lift"][i]) == 'nan':
                self.dictionary_sreality["has_lift"][i] = None
            self.dictionary_sreality["has_lift"][i] = bool(self.dictionary_sreality["has_lift"][i])
            self.dictionary_sreality["has_lift"][i] = float(self.dictionary_sreality["has_lift"][i])
        print(set(self.dictionary_sreality["has_lift"].values()))

        # has cellar
        for i in range(len(self.dictionary_sreality["has_cellar"])):
            if str(self.dictionary_sreality["has_cellar"][i]) == 'nan':
                self.dictionary_sreality["has_cellar"][i] = None
            self.dictionary_sreality["has_cellar"][i] = bool(self.dictionary_sreality["has_cellar"][i])
            self.dictionary_sreality["has_cellar"][i] = float(self.dictionary_sreality["has_cellar"][i])

        # has loggia
        for i in range(len(self.dictionary_sreality["has_loggia"])):
            if str(self.dictionary_sreality["has_loggia"][i]) == 'nan':
                self.dictionary_sreality["has_loggia"][i] = None
            self.dictionary_sreality["has_loggia"][i] = bool(self.dictionary_sreality["has_loggia"][i])
            self.dictionary_sreality["has_loggia"][i] = float(self.dictionary_sreality["has_loggia"][i])

        # has balcony
        for i in range(len(self.dictionary_sreality["has_balcony"])):
            if str(self.dictionary_sreality["has_balcony"][i]) == 'nan':
                self.dictionary_sreality["has_balcony"][i] = None
            self.dictionary_sreality["has_balcony"][i] = bool(self.dictionary_sreality["has_balcony"][i])
            self.dictionary_sreality["has_balcony"][i] = float(self.dictionary_sreality["has_balcony"][i])

        # has parking
        for i in range(len(self.dictionary_sreality["has_parking"])):
            if str(self.dictionary_sreality["has_parking"][i]) == 'nan':
                self.dictionary_sreality["has_parking"][i] = None
            self.dictionary_sreality["has_parking"][i] = bool(self.dictionary_sreality["has_parking"][i])
            self.dictionary_sreality["has_parking"][i] = float(self.dictionary_sreality["has_parking"][i])



        return self.dictionary_sreality

# sreality_df = pd.read_csv("C:/Users/adams/OneDrive/Dokumenty/GitHub/real_estate/data/prodej_sreality_scraped.csv")
synchronizer = Synchronizer(tuple([0,0]))
synchronizer.__call__(sreality_csv_path="C:/Users/adams/OneDrive/Dokumenty/GitHub/real_estate/data/prodej_sreality_scraped.csv", breality_csv_path="C:/Users/adams/OneDrive/Dokumenty/GitHub/real_estate/data/prodej_breality_scraped.csv")
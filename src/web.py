# documentation: https://docs.streamlit.io/
# more: https://extras.streamlitapp.com/Altex
import base64
import csv
# pip install streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import os
from streamlit_folium import folium_static
import folium
import streamlit as st
from datetime import datetime
import joblib
import pickle
import py7zr
from folium.plugins import MousePosition
import requests

from main import ETL, Model
from models.gaussian_process import get_gp

st.set_page_config(page_title='Real e-state', page_icon="house_buildings", initial_sidebar_state="collapsed")

# Navigation menu from: https://github.com/Sven-Bo/streamlit-navigation-menu
# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 3
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Projects", "Contact"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Domů", "Predikce pomocí URL", "Predikce pomocí ručně zadaných příznaků", "Kontakt"],  # required
            icons=["house", "graph-down", "graph-up", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "grey"},
            },
        )
        return selected

def get_csv_handmade():
    # type
    type = st.radio("Typ", (
        '1+kk', '1+1', '2+kk', '2+1', '3+kk', '3+1', '4+kk', '4+1', '5+kk', '5+1', '6', '6+kk', 'atypické'))
    disposition_dict = None
    if type == '1+kk':
        disposition_dict = '1+kk'
    elif type == '1+1':
        disposition_dict = '1+1'
    elif type == '2+kk':
        disposition_dict = '2+kk'
    elif type == '2+1':
        disposition_dict = '2+1'
    elif type == '3+kk':
        disposition_dict = '3+kk'
    elif type == '3+1':
        disposition_dict = '3+1'
    elif type == '4+kk':
        disposition_dict = '4+kk'
    elif type == '4+1':
        disposition_dict = '1+kk4+1'
    elif type == '5+kk':
        disposition_dict = '5+kk'
    elif type == '5+1':
        disposition_dict = '5+1'
    elif type == '6':
        disposition_dict = '6'
    elif type == '6+kk':
        disposition_dict = '6+kk'
    elif type == 'atypické':
        disposition_dict = 'atypické'
    else:
        disposition_dict = np.NaN

    # usable area
    usable_area = st.number_input('Užitná plocha v m^2', step=1)
    usable_area_dict = None
    if usable_area <= 0:
        print('error usable area must be positive!')
        usable_area_dict = None
    else:
        usable_area_dict = usable_area  # využijeme text pro model

    # energy eficiency
    energy = st.radio("Energetická eficience", ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
    energy_dict = None
    if energy == 'A':
        energy_dict = 'A'
    elif energy == 'B':
        energy_dict = 'B'
    elif energy == 'C':
        energy_dict = 'C'
    elif energy == 'D':
        energy_dict = 'D'
    elif energy == 'E':
        energy_dict = 'E'
    elif energy == 'F':
        energy_dict = 'F'
    elif energy == 'G':
        energy_dict = 'G'
    else:
        energy_dict = np.NaN

    # floor
    floor = st.number_input('Patro', step=1)
    floor_dict = None
    if floor < -1:
        print('error in floor! must be higher than -1')
        floor_dict = None
    else:
        floor_dict = floor  # využijeme text pro model

    col1, col2, col3 = st.columns(3)
    with col1:
        ownership = st.radio("Vlastnictví", ('Osobní', 'Státní/obecní', 'Družstevní'))
        ownership_dict = None
        if ownership == 'Osobní':
            ownership_dict = 'Osobní'
        elif ownership == 'Státní/obecní':
            ownership_dict = 'Státní/obecní'
        elif ownership == 'Družstevní':
            ownership_dict = 'Družstevní'
        else:
            ownership_dict = np.NaN

    with col2:
        equipment_dict = None
        equipment = st.radio("Vybavenost", ('Plně', 'Nevybaveno', 'Částečně'))
        if equipment == 'Plně':
            equipment_dict = 'Plně'
        elif equipment == 'Nevybaveno':
            equipment_dict = 'Nevybaveno'
        elif equipment == 'Částečně':
            equipment_dict = 'Částečně'
        else:
            equipment_dict = np.NaN

    with col3:
        podkrovni = st.checkbox('Podkrovní')
        podkrovni_dict = None
        loft = st.checkbox('Loft')
        loft_dict = None
        mezonet = st.checkbox('Mezonet')
        mezonet_dict = None
        if podkrovni:
            podkrovni_dict = True
        if loft:
            loft_dict = True
        if mezonet:
            mezonet = True

    with col1:
        state = st.radio("Stav", ('V rekonstrukci', 'Před rekonstrukcí', 'Po rekonstrukci', 'Nová budova',
                                  'Velmi dobrý', 'Dobrý', 'Staví se', 'Projekt', 'Špatný'))
    with col2:
        construction = st.radio("Konstrukce", (
        'Cihlová', 'Smíšená', 'Panelová', 'Skeletová', 'Kamenná', 'Montovaná', 'Nízkoenergetická'))

    # state
    state_dict = None
    if state == 'V rekonstrukci':
        state_dict = 'V rekonstrukci'
    elif state == 'Před rekonstrukcí':
        state_dict = 'Před rekonstrukcí'
    elif state == 'Po rekonstrukci':
        state_dict = 'Po rekonstrukci'
    elif state == 'Nová budova':
        state_dict = 'Nová budova'
    elif state == 'Velmi dobrý':
        state_dict = 'Velmi dobrý'
    elif state == 'Dobrý':
        state_dict = 'Dobrý'
    elif state == 'Staví se':
        state_dict = 'Staví se'
    elif state == 'Projekt':
        state_dict = 'Projekt'
    elif state == 'Špatný':
        state_dict = 'Špatný'
    else:
        state_dict = np.NaN

    # construction
    construction_dict = None
    if construction == 'Cihlová':
        construction_dict = 'Cihlová'
    elif construction == 'Smíšená':
        construction_dict = 'Smíšená'
    elif construction == 'Panelová':
        construction_dict = 'Panelová'
    elif construction == 'Skeletová':
        construction_dict = 'Skeletová'
    elif construction == 'Kamenná':
        construction_dict = 'Kamenná'
    elif construction == 'Montovaná':
        construction_dict = 'Montovaná'
    elif construction == 'Nízkoenergetická':
        construction_dict = 'Nízkoenergetická'
    else:
        construction_dict = np.NaN

    # balcony, terrace, parking, lift, loggia, cellar, garage, garden
    col4, col5, col6 = st.columns(3)
    with col4:
        balcony = st.checkbox('Má balkón')
        balcony_dict = None
        if balcony:
            balcony_dict = True
        terrace = st.checkbox('Má terasu')
        terrace_dict = None
        if terrace:
            terrace_dict = True
        parking = st.checkbox('Má prakování (venkovní)')
        parking_dict = None
        if parking:
            parking_dict = True
        lift = st.checkbox('Má výtah')
        lift_dict = None
        if lift:
            lift_dict = True

    with col5:
        loggia = st.checkbox('Má lodžie')
        loggia_dict = None
        if loggia:
            loggia_dict = True
        cellar = st.checkbox('Má sklep')
        cellar_dict = None
        if cellar:
            cellar_dict = True
        garage = st.checkbox('Má garáž')
        garage_dict = None
        if garage:
            garage_dict = True
        garden = st.checkbox('Má zahradu')
        garden_dict = None
        if garden:
            garden_dict = True

    # TODO GPS - map: https://discuss.streamlit.io/t/ann-streamlit-folium-a-component-for-rendering-folium-maps/4367/4
    x = st.number_input('GPS N - lattitude')
    y = st.number_input('GPS E - longtitude')
    # starting point
    if x == 0:
        x = 50.0818633
    if y == 0:
        y = 14.4255628
    # 49.2107581, 16.6188150
    m = folium.Map(location=[x, y], zoom_start=10)
    folium.LatLngPopup().add_to(m)
    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker([x, y], tooltip=tooltip).add_to(m)
    # call to render Folium map in Streamlit
    folium_static(m)
    # save data
    out = {
        'header': None,
        # text description of disposition e.g. 3 + kk
        'price': None,  # Celková cena
        'note': None,
        'usable_area': usable_area_dict,
        'floor_area': None,
        'floor': floor_dict,
        'energy_effeciency': energy_dict,
        'ownership': ownership_dict,
        # vlastnictvo (3 possible) vlastni/druzstevni/statni(obecni)
        'description': None,
        'long': y,
        'lat': x,
        'hash': None,

        # other - done
        'gas': None,  # Plyn
        'waste': None,  # Odpad
        'equipment': equipment_dict,  # Vybavení
        'state': state_dict,
        'construction_type': construction_dict,
        'place': None,  # Umístění objektu
        'electricity': None,  # elektrina
        'heating': None,  # topeni
        'transport': None,  # doprava
        'year_reconstruction': None,  # rok rekonstrukce
        'telecomunication': None,  # telekomunikace

        # binary info - done
        'has_lift': lift_dict,  # Výtah: True, False
        'has_garage': garage_dict,  # garaz
        'has_cellar': cellar_dict,  # sklep presence
        'no_barriers': None,  # ci je bezbarierovy bezbarierovy
        'has_loggia': loggia_dict,  # lodzie
        'has_balcony': balcony_dict,  # balkon
        'has_garden': garden_dict,  # zahrada
        'has_parking': parking_dict,

        # what has b reality in addition
        'tags': None,
        'disposition': disposition_dict,

        # closest distance to civic amenities (in metres) (obcanska vybavenost vzdialenosti) -
        'bus_station_dist': None,
        'train_station_dist': None,
        'subway_station_dist': None,
        'tram_station_dist': None,
        'post_office_dist': None,
        'atm_dist': None,
        'doctor_dist': None,
        'vet_dist': None,
        'primary_school_dist': None,
        'kindergarten_dist': None,
        'supermarket_grocery_dist': None,
        'restaurant_pub_dist': None,
        'playground_dist': None,
        'sports_field_dist': None,
        # or similar kind of leisure amenity probably OSM would be better
        # 'park': None -- probably not present => maybe can be within playground or we will scrape from OSM
        'theatre_cinema_dist': None,
        'pharmacy_dist': None,
        'name': None,
        'date': datetime.today().strftime('%Y-%m-%d')

    }
    return out

selected = streamlit_menu(example=EXAMPLE_NO)

############## 1. stránka ##############
if selected == "Domů":
    st.header(f"Real e-state")
    st.markdown(":sparkles: Naše vize je pomoci lidem predikovat ceny nemovitostí (bytů v Praze). Predikovat lze pomocí zadaného "
                 "URL, z sreality.cz nebo bezrealitky.cz, nebo pomocí ručně zadaných vlastností. Dále můžeme investorům pomoci detekovat, "
              "jaké nemovitosti na trhu jsou podceněné nebo nadceněné a do kterých je lepší investovat. "
              "Bonusem bude dodání dalších informací o nemovitosti.")

############## 2. stránka ##############
if selected == "Predikce pomocí URL":
    st.header("Predikce ceny nemovitosti pomocí URL")
    # url
    url = st.text_input('URL nemovitosti (bytu v Praze) z sreality.cz or bezrealitky.cz')
    if url is None:
        pass
    else:
        url = str(url)
        if 'bezrealitky' or 'sreality' in url:
            with open('../data/predict_links.txt', 'w') as f:
                f.write(url)

    ############## MODELS ##############
    result_url = st.button('Predikuj!')
    if result_url:
        
        st.markdown(f'Získávám data z {url}...')
        st.markdown(':robot_face: Robot přemýšlí...')
        etl = ETL(inference=True)
        out = etl()

        if out['status'] != 'OK':
            st.write(f'Užitná plocha, zemepisna sirka a vyska su povinne atributy')
        else:
            ## just for now only fitted gaussian process is used

            model_path = 'models/fitted_gp_low'
            gp_model = get_gp(model_path)
    
            X = out['data'][['long', 'lat']].to_numpy()
            mean_price, std_price = gp_model.predict(X, return_std=True)
            price_gp = (mean_price * out['data']["usable_area"].to_numpy()).item()
            std = (std_price * out['data']["usable_area"].to_numpy()).item()
            st.write(f'Cena bytu podla GP je {price_gp} Kc. \n' 
                     f'95% konfidencni interval {(price_gp - 2 * std, price_gp + 2 * std)} Kc')

            # OTHER MODELS
            model = Model(data=out['data'], inference=True, tune=False, response='log_price')
            price = model()
            st.write(f'Cena tohto bytu je odhadnuta podla XGB na {price.item()} Kc. \n')


if selected == "Predikce pomocí ručně zadaných příznaků":
    st.header(f"Predikce pomocí ručně zadaných příznaků")
    out = get_csv_handmade()
    field_names = []
    for key, value in out.items():
        field_names.append(key)

    ############## MODELS ##############
    result = st.button('Predikuj!')
    ''' if result:
        # st.write('Calculating results...')
        # RANDOM FOREST
        # regr = RandomForestRegressor(n_estimators=350, max_depth=7, max_features=10, min_samples_leaf=3, criterion='absolute_error', random_state=42)
        loaded_rf = joblib.load("./random_forest.joblib") # https://mljar.com/blog/save-load-random-forest/
        rf_price = loaded_rf.predict(X)
        st.write('Predicting price of the flat is: ', rf_price)'''

    if result:
        with open('../data/predict_handmade.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows([out])
            print('csv done!')
        st.markdown(':robot_face: Robot přemýšlí...')



############## 3. stránka ##############
if selected == "Kontakt":
    st.header(f"Kontakt")
    st.markdown(":copyright: Zkoukněte náš [GitHub](https://github.com/Many98/real_estate).")

# background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('../data/misc/houses0.jpg')

# hide icon streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


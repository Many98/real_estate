# documentation: https://docs.streamlit.io/
# more: https://extras.streamlitapp.com/Altex
import base64
# pip install streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import os
from streamlit_folium import folium_static
import folium
import streamlit as st
import joblib
import pickle
import py7zr
from folium.plugins import MousePosition
import requests

from main import ETL
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

    # MODELS
    result_url = st.button('Predikuj!')
    if result_url:
        
        st.markdown(f'Získávám data z {url}...')
        st.markdown(':robot_face: Robot přemýšlí...')
        etl = ETL(inference=True)
        data = etl()
        
        ## just for now only fitted gaussian process is used
        model_path = 'models/fitted_gp_low'
        gp_model = get_gp(model_path)

        X = data[['long', 'lat']].to_numpy()
        mean_price, std_price = gp_model.predict(X, return_std=True)
        price = mean_price * data["usable_area"].to_numpy()
        std = std_price * data["usable_area"].to_numpy()
        st.write(f'Estimated price of apartment is {price} Kc. \n' 
                 f'95% confidece interval is {(price - 2 * std, price + 2 * std)} Kc')

        # OTHER MODELS


if selected == "Predikce pomocí ručně zadaných příznaků":
    st.header(f"Predikce pomocí ručně zadaných příznaků")
    # type
    type = st.radio("Typ", (
    '1+kk', '1+1', '2+kk', '2+1', '3+kk', '3+1', '4+kk', '4+1', '5+kk', '5+1', '6', '6+kk', 'atypické'))
    if type == '1+kk':
        c = 1
    elif type == '1+1':
        c = 1
    elif type == '2+kk':
        c = 1
    elif type == '2+1':
        c = 1
    elif type == '3+kk':
        c = 1
    elif type == '3+1':
        c = 1
    elif type == '4+kk':
        c = 1
    elif type == '4+1':
        c = 1
    elif type == '5+kk':
        c = 1
    elif type == '5+1':
        c = 1
    elif type == '6':
        c = 1
    elif type == '6+kk':
        c = 1
    elif type == 'atypické':
        c = 1
    else:
        c = np.NaN

    # usable area
    usable_area = st.number_input('Užitná plocha v m^2', step = 1)
    if usable_area == 0:
        pass
    else:
        b = 1  # využijeme text pro model

    # energy eficiency
    energy = st.radio("Energetická eficience", ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
    if energy == 'A':
        d = 1
    elif energy == 'B':
        d = 1
    elif energy == 'C':
        d = 1
    elif energy == 'D':
        d = 1
    elif energy == 'E':
        d = 1
    elif energy == 'F':
        d = 1
    elif energy == 'G':
        d = 1
    else:
        d = np.NaN

    # floor
    floor = st.number_input('Patro', step = 1)
    if floor == 0:
        pass
    else:
        f = 1 # využijeme text pro model

    col1, col2, col3 = st.columns(3)
    with col1:
        ownership = st.radio("Vlastnictví", ('Osobní', 'Státní/obecní', 'Družstevní'))
    with col2:
        equipment = st.radio("Vybavenost", ('Plně', 'Nevybaveno', 'Částečně'))
    with col3:
        podkrovni = st.checkbox('Podkrovní')
        loft = st.checkbox('Loft')
        mezonet = st.checkbox('Mezonet')
        if podkrovni:
            pass
        if loft:
            pass
        if mezonet:
            pass

    # ownership
    if ownership == 'Osobní':
        e = 1
    elif ownership == 'Státní/obecní':
        e = 1
    elif ownership == 'Družstevní':
        e = 1
    else:
        e = np.NaN

    # equipment
    if equipment == 'Plně':
        g = 1
    elif equipment == 'Nevybaveno':
        g = 1
    elif equipment == 'Částečně':
        g = 1
    else:
        g = np.NaN

    with col1:
        state = st.radio("Stav", ('V rekonstrukci', 'Před rekonstrukcí', 'Po rekonstrukci', 'Nová budova',
                                  'Velmi dobrý', 'Dobrý', 'Staví se', 'Projekt', 'Špatný'))
    with col2:
        construction = st.radio("Konstrukce", ('Cihlová', 'Smíšená', 'Panelová', 'Skeletová', 'Kamenná', 'Montovaná', 'Nízkoenergetická'))

    # state
    if state == 'V rekonstrukci':
        h = 1
    elif state == 'Před rekonstrukcí':
        h = 1
    elif state == 'Po rekonstrukci':
        h = 1
    elif state == 'Nová budova':
        h = 1
    elif state == 'Velmi dobrý':
        h = 1
    elif state == 'Dobrý':
        h = 1
    elif state == 'Staví se':
        h = 1
    elif state == 'Projekt':
        h = 1
    elif state == 'Špatný':
        h = 1
    else:
        h = np.NaN

    # construction TODO

    # TODO (?) 'gas', 'electricity', 'waste', 'heating', 'telecomunication'

    # balcony, terrace, parking, lift, loggia, cellar, garage, garden
    col4, col5, col6 = st.columns(3)
    with col4:
        balcony = st.checkbox('Má balkón')
        if balcony:
            pass
        terrace = st.checkbox('Má terasu')
        if terrace:
            pass
        parking = st.checkbox('Má prakování (venkovní)')
        if parking:
            pass
        lift = st.checkbox('Má výtah')
        if lift:
            pass
    with col5:
        loggia = st.checkbox('Má lodžie')
        if loggia:
            pass
        cellar = st.checkbox('Má sklep')
        if cellar:
            pass
        garage = st.checkbox('Má garáž')
        if garage:
            pass
        garden = st.checkbox('Má zahradu')
        if garden:
            pass

    # TODO GPS - map: https://discuss.streamlit.io/t/ann-streamlit-folium-a-component-for-rendering-folium-maps/4367/4
    x = st.number_input('GPS N')
    y = st.number_input('GPS E')
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
    #save data

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


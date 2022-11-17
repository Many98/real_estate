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
import requests

from main import ETL

st.set_page_config(page_title='Real e-state', page_icon="chart_with_upwards_trend", initial_sidebar_state="collapsed")

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
            options=["Home", "Prediction by URL", "Handmade prediction", "Contact"],  # required
            icons=["house", "book", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "20px"},
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
if selected == "Home":
    st.title(f"Real e-state")
    st.header("This application is able to predict a price of property in Prague. "
              "Prediction using parametres of the property, description, images and GPS.")

############## 2. stránka ##############
if selected == "Prediction by URL":
    st.title(f"Price prediction of property with URL")
    # url
    url = st.text_input('URL of property on sreality or bezrealitky')
    if url is None:
        pass
    else:
        url = str(url)
        if 'bezrealitky' or 'sreality' in url:
            with open('../data/predict_links.txt', 'w') as f:
                f.write(url)

    # MODELS
    result_url = st.button('Predict house price with URL')
    if result_url:
        
        st.write(f'Scraping data from {result_url} ...')
        
        etl = ETL(inference=True)
        data = etl()
        
        ## just for now only fitted gaussian process is used
        model_path = 'models/fitted_gp_low'
        if not os.path.isfile(model_path):
            with requests.get('https://zenodo.org/record/7319710/files/fitted_gp_low.7z?download=1', stream=True) as r:
                with open('models/fitted_gp_low.7z', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        if '7z' in model_path:
            if not os.path.isfile(os.path.split(model_path)[0]):
                with py7zr.SevenZipFile(model_path, mode='r') as z:
                    z.extractall(path=os.path.split(model_path)[0])
            model_path = os.path.split(model_path)[0]
        if os.path.isfile(model_path):
            gp_model = pickle.load(open(model_path, 'rb'))
        else:
            raise Exception('model not found')
         
        st.write('Model is thinking...')
        X = data[['long', 'lat']].to_numpy()
        mean_price, std_price = gp_model.predict(X, return_std=True)
        
        st.write(f'Estimated price of apartment is {mean_price}. \n' 
                 f'95% confidece interval is {(mean_price - 2 * std_price, mean_price + 2 * std_price)}')

        # OTHER MODELS


if selected == "Handmade prediction":
    st.title(f"Price prediction of property with handmade features")
    # type
    type = st.radio("Type of flat", (
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
    usable_area = st.number_input('Usable area in m^2', step = 1)
    if usable_area == 0:
        pass
    else:
        b = 1  # využijeme text pro model

    # energy eficiency
    energy = st.radio("Energy efficiency", ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
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
    floor = st.number_input('Floor', step = 1)
    if floor == 0:
        pass
    else:
        f = 1 # využijeme text pro model

    # description of your house
    text = st.text_input('Write a description of your house in few sentences')
    if text is None:
        pass
    else:
        a = 1 # využijeme text pro model

    col1, col2, col3 = st.columns(3)
    with col1:
        ownership = st.radio("Ownership", ('Osobní', 'Státní/obecní', 'Družstevní'))
    with col2:
        equipment = st.radio("Equipment", ('Yes', 'No', 'Partly'))
    with col3:
        addictional = st.radio("Addictional dispositon", ('Studio apartment', 'Loft', 'Mezonet'))

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
    if equipment == 'Yes':
        g = 1
    elif equipment == 'No':
        g = 1
    elif equipment == 'Partly':
        g = 1
    else:
        g = np.NaN

    # addictional dispositon
    if addictional == 'Studio apartment':
        i = 1
    elif addictional == 'Loft':
        i = 1
    elif addictional == 'Mezonet':
        i = 1
    else:
        i = np.NaN


    with col1:
        state = st.radio("State", ('In reconstruction', 'Before reconstruction', 'After reconstruction', 'New building', 'Very good', 'Good', 'Building-up', 'Project', 'Bad'))
    with col2:
        construction = st.radio("Construction type", ('Cihlová', 'Smíšená', 'Panelová', 'Skeletová', 'Kamenná', 'Montovaná', 'Nízkoenergetická'))

    # state
    if state == 'In reconstruction':
        h = 1
    elif state == 'Before reconstruction':
        h = 1
    elif state == 'After reconstruction':
        h = 1
    elif state == 'New building':
        h = 1
    elif state == 'Very good':
        h = 1
    elif state == 'Good':
        h = 1
    elif state == 'Building-up':
        h = 1
    elif state == 'Project':
        h = 1
    elif state == 'Bad':
        h = 1
    else:
        h = np.NaN

    # construction TODO

    # TODO (?) 'gas', 'electricity', 'waste', 'heating', 'telecomunication'

    # balcony, terrace, parking, lift, loggia, cellar, garage, garden
    col4, col5, col6 = st.columns(3)
    with col4:
        balcony = st.checkbox('Has balcony')
        if balcony:
            pass
        terrace = st.checkbox('Has terrace')
        if terrace:
            pass
        parking = st.checkbox('Has parking')
        if parking:
            pass
        lift = st.checkbox('Has lift')
        if lift:
            pass
    with col5:
        loggia = st.checkbox('Has loggia')
        if loggia:
            pass
        cellar = st.checkbox('Has cellar')
        if cellar:
            pass
        garage = st.checkbox('Has garage')
        if garage:
            pass
        garden = st.checkbox('Has garden')
        if garden:
            pass

    # TODO GPS - map: https://discuss.streamlit.io/t/ann-streamlit-folium-a-component-for-rendering-folium-maps/4367/4
    x = st.number_input('GPS N')
    y = st.number_input('GPS E')
    # 49.2107581, 16.6188150
    m = folium.Map(location=[x, y], zoom_start=16)
    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker(
        [x, y], tooltip=tooltip
    ).add_to(m)
    # call to render Folium map in Streamlit
    folium_static(m)
    #save data
    ############## MODELS ##############
    result = st.button('Predict house price with handmade inputs')
    ''' if result:
        # st.write('Calculating results...')
        # RANDOM FOREST
        # regr = RandomForestRegressor(n_estimators=350, max_depth=7, max_features=10, min_samples_leaf=3, criterion='absolute_error', random_state=42)
        loaded_rf = joblib.load("./random_forest.joblib") # https://mljar.com/blog/save-load-random-forest/
        rf_price = loaded_rf.predict(X)
        st.write('Predicting price of the flat is: ', rf_price)'''

        #OTHER MODELS






############## 3. stránka ##############
if selected == "Contact":
    st.title(f"Contact")
    st.markdown("Check out our repository on GitHub [link](https://github.com/Many98/real_estate).")

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


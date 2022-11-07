# documentation: https://docs.streamlit.io/
# more: https://extras.streamlitapp.com/Altex
import streamlit as st
import base64
import streamlit as st
# pip install streamlit_option_menu
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import io
import os
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import folium
import requests
from requests.exceptions import ConnectionError
import streamlit as st
from folium.plugins import Draw
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import pickle  # to load a saved model

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
            options=["Home", "Prediction", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
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

if selected == "Home":
    st.title(f"Real e-state")
    st.header("This application is able to predict a price of property in Prague. "
              "Prediction using parametres of the property, description, images and GPS.")

if selected == "Prediction":
    st.title(f"Price prediction of property")
    # photo of your property
    uploaded_file = st.file_uploader('Upload a photo of your property')
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # BytesIO object to array
        buf = io.BytesIO(bytes_data)
        img = Image.open(buf)
        img = np.asarray(img) # využijeme obrázek pro model

    # description of your house
    text = st.text_input('Write a description of your house in few sentences')
    if text is None:
        pass
    else:
        a = 1 # využijeme text pro model

    # GPS - map: https://discuss.streamlit.io/t/ann-streamlit-folium-a-component-for-rendering-folium-maps/4367/4
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
add_bg_from_local('houses0.jpg')

# hide icon streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


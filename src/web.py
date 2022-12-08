# documentation: https://docs.streamlit.io/
# more: https://extras.streamlitapp.com/Altex
import base64
import csv
# pip install streamlit_option_menu
import streamlit_echarts
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import folium
import pandas as pd
import streamlit as st
from datetime import datetime
from pyecharts import options as opts
from pyecharts.charts import Bar
import altair as alt
from streamlit_echarts import st_pyecharts
from streamlit_echarts import st_echarts
import joblib
import pickle
import py7zr
from folium.plugins import MousePosition
import requests
import locale
from streamlit_shap import st_shap
import shap

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


def get_pos(lat, lng):
    return lat, lng


def get_csv_handmade():
    # type
    # TODO - None type for st.radio
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
    # usable_area = st.number_input('Užitná plocha v m^2', step=1)
    usable_area = st.slider('Užitná plocha v m^2', 0, 1000)
    usable_area_dict = None
    if usable_area <= 0:
        print('error usable area must be positive!')
        usable_area_dict = None
    else:
        usable_area_dict = usable_area  # využijeme text pro model

    # energy eficiency
    energy = st.select_slider(
        'Energetická eficience',
        options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    # energy = st.radio("Energetická eficience", ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
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
    # floor = st.number_input('Patro', step=1)
    floor = st.slider('Patro', -1, 20)
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
            equipment_dict = 'ano'
        elif equipment == 'Nevybaveno':
            equipment_dict = 'Nevybaveno'
        elif equipment == 'Částečně':
            equipment_dict = 'ne'
        else:
            equipment_dict = np.NaN

    with col1:
        state = st.radio("Stav", ('V rekonstrukci', 'Před rekonstrukcí', 'Po rekonstrukci', 'Nová budova',
                                  'Velmi dobrý', 'Dobrý', 'Staví se', 'Projekt', 'Špatný'))
    with col2:
        construction = st.radio("Konstrukce", (
            'Cihlová', 'Smíšená', 'Panelová', 'Skeletová', 'Kamenná', 'Montovaná', 'Nízkoenergetická', 'Drevostavba'))

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
    elif construction == 'Drevostavba':
        construction_dict = 'Drevostavba'
    else:
        construction_dict = np.NaN

    # balcony, terrace, parking, lift, loggia, cellar, garage, garden
    col4, col5, col6 = st.columns(3)
    with col4:
        balcony = st.checkbox('Má balkón')
        balcony_dict = False
        if balcony:
            balcony_dict = True
        terrace = st.checkbox('Má terasu')
        terrace_dict = False
        if terrace:
            terrace_dict = True
        parking = st.checkbox('Má parkování (venkovní)')
        parking_dict = False
        if parking:
            parking_dict = True
        lift = st.checkbox('Má výtah')
        lift_dict = False
        if lift:
            lift_dict = True

    with col5:
        loggia = st.checkbox('Má lodžie')
        loggia_dict = False
        if loggia:
            loggia_dict = True
        cellar = st.checkbox('Má sklep')
        cellar_dict = False
        if cellar:
            cellar_dict = True
        garage = st.checkbox('Má garáž')
        garage_dict = False
        if garage:
            garage_dict = True
        garden = st.checkbox('Má zahradu')
        garden_dict = False
        if garden:
            garden_dict = True

    # TODO GPS - map: https://discuss.streamlit.io/t/ann-streamlit-folium-a-component-for-rendering-folium-maps/4367/4
    # x = st.number_input('GPS N - lattitude')
    # y = st.number_input('GPS E - longtitude')
    # starting point
    x = 50.0818633
    y = 14.4255628
    m = folium.Map(location=[x, y], zoom_start=10)

    m.add_child(folium.LatLngPopup())
    map = st_folium(m, height=350, width=700)
    if map['last_clicked'] is None:
        lat, long = 55, 12
    else:
        lat, long = get_pos(map['last_clicked']['lat'], map['last_clicked']['lng'])
    x = lat
    y = long

    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker([x, y], tooltip=tooltip).add_to(m)
    print(x, y)

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
        'description': 'none',
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


def render_noise_gauge(noise):
    pass
    # https://echarts.apache.org/examples/en/editor.html?c=gauge-grade


def render_bar_plot_v2():
    # https://echarts.apache.org/examples/en/editor.html?c=bar-negative2
    option = {
        "title": {
            "text": 'Efekty jednotlivých atributu bytu na cenu',
            "left": 'center'
        },
        "tooltip": {
            "trigger": 'axis',
            "axisPointer": {
                "type": 'shadow'
            }
        },
        "grid": {
            "top": 80,
            "bottom": 30
        },
        "xAxis": {
            "type": 'value',
            "position": 'top',
            "splitLine": {
                "lineStyle": {
                    "type": 'dashed'
                }
            }
        },
        "yAxis": {
            "type": "category",
            "show": False,
            "data": [  # TODO here comes the feature names + values
                'Prumerna cena bytu v oblasti',
                'Uzitna plocha: 55 m2',
                'Standardni odchylka ceny bytu',
                'disposition: 4+kk',
                'Mestska cast: Praha 4',
                '38 dalsich priznaku',

            ]
        },
        "series": [
            {
                "name": 'Efekt',
                "type": 'bar',
                "stack": 'Total',
                "label": {
                    "show": True,
                    "textBorderColor": 'white',
                    "color": 'black',
                    "textBorderWidth": 8,
                    "formatter": '{c} Kč'
                }
                ,
                "data": [  # TODO here comes data about shap values

                    -100000,
                    140000,
                    6000,
                    7000,
                    80000,
                    25000,

                ]
            }
        ],
        "visualMap": {
            "orient": 'horizontal',
            "left": 'center',
            "min": 0,
            "max": 1,
            "text": ['Kladný efekt', 'Záporný efekt'],

            "dimension": 0,
            "inRange": {
                "color": ['red', 'green']
            }
        },
    }

    st_echarts(option, height="700px", key="echarts_bar2")


def render_donut_plot():
    # https://echarts.apache.org/examples/en/editor.html?c=pie-doughnut
    option = {
        "title": {
            "text": 'Kriminalita v okolí',
            "left": 'center'
        },

        "legend": {
            "top": '5%',
            "left": 'center'
        },
        "tooltip": {},
        "series": [
            {
                "name": 'Zločin',
                "type": 'pie',
                "radius": ['40%', '70%'],
                "avoidLabelOverlap": "false",

                "data": [  # TODO here comes custom data
                    {"value": 8, "name": 'Krádež'},
                    {"value": 73, "name": 'Vlopání'},
                    {"value": 58, "name": 'Nehoda'},
                    {"value": 48, "name": 'únos'},
                    {"value": 3, "name": 'Vražda'}
                ]
            }
        ]
    }

    st_echarts(option, height="500px", key="echarts_donut")


def render_ring_gauge_quality(sun, air, built):
    # https://echarts.apache.org/examples/en/editor.html?c=gauge-ring
    option = {
        "title": [
            {
                "text": 'Kvalitativní data',
                "left": 'center'
            }
        ],
        "tooltip": {},
        "series": [
            {
                "type": "gauge",
                "startAngle": 90,
                "endAngle": -250,
                "pointer": {"show": False},
                "progress": {
                    "show": True,
                    "overlap": False,
                    "roundCap": True,
                    "clip": False,
                    "itemStyle": {"borderWidth": 1, "borderColor": "#464646"},
                },
                "axisLine": {"lineStyle": {"width": 40}},
                "splitLine": {"show": False, "distance": 0, "length": 10},
                "axisTick": {"show": False},
                "axisLabel": {"show": False, "distance": 50},
                "data": [
                    {
                        "value": air,
                        "name": "🌪️ Kvalita vzduchu",
                        "title": {"offsetCenter": ["0%", "-30%"]},
                        "detail": {"offsetCenter": ["0%", "-20%"]},
                    },
                    {
                        "value": built,
                        "name": "👫 Úroveň zastavěnosti",
                        "title": {"offsetCenter": ["0%", "0%"]},
                        "detail": {"offsetCenter": ["0%", "10%"]},
                    },
                    {
                        "value": sun,
                        "name": "🌞 Kvalita oslunění",
                        "title": {"offsetCenter": ["0%", "30%"]},
                        "detail": {"offsetCenter": ["0%", "40%"]},
                    },

                ],
                "title": {"fontSize": 11},
                "detail": {
                    "width": 100,
                    "height": 6,
                    "fontSize": 12,
                    "color": "auto",
                    "borderColor": "auto",
                    "borderRadius": 20,
                    "borderWidth": 1,
                    "formatter": streamlit_echarts.JsCode(
                        'function lambda(a){return 20===a?"Velmi nizka":40===a?"Nizka":60===a?"Stredni":80===a?"Vysoka":100===a?"Velmi vysoka":"Data nedostupna"}').js_code,
                },
            }
        ]
    }
    st_echarts(option, height="500px", key="echarts_gauge")


def render_bar_prediction(lower, mean, upper):
    # https://echarts.apache.org/examples/en/editor.html?c=dataset-encode0
    option = {
        "title": [
            {
                "text": 'Odhad ceny bytu',
                "left": 'center'
            }
        ],
        "dataset": {
            "source": [
                ['score', 'amount', 'product'],
                [lower, lower, '5% kvantil'],
                [mean, mean, 'Predikce'],
                [upper, upper, '95% kvantil'],

            ]
        },
        "grid": {"containLabel": "true"},
        "xAxis": {"name": 'Cena v Kč'},
        "yAxis": {"type": 'category'},
        "visualMap": {
            "orient": 'horizontal',
            "left": 'center',
            "min": min(lower, mean),
            "max": max(upper, mean),
            "text": ['95% kvantil' if upper > mean else 'Predikce', '5% kvantil' if lower < mean else 'Predikce'],
            "dimension": 0,
            "inRange": {
                "color": ['#65B581', '#FFCE34', '#FD665F']
            }
        },
        "tooltip": {},
        "series": [
            {
                "type": "bar",
                "encode": {

                    "x": "amount",

                    "y": "product"
                }
            }
        ]
    }

    st_echarts(option, height="500px", key="echarts_prediction")


def prediction(handmade, url=''):
    with st.spinner(':robot_face: Robot přemýšlí...'):

        etl = ETL(inference=True, handmade=handmade)
        out = etl()

    if out['status'] == 'RANP':
        st.warning('Užitná plocha, zeměpisna šířka a výška jsou povinné atributy', icon="⚠️")

        # st.write(f'Užitná plocha, zemepisna sirka a vyska su povinne atributy')
    elif out['status'] == 'EMPTY':
        # st.write(f'Data nejsou k dispozici')
        st.warning('Data nejsou k dispozici', icon="⚠️")
    elif 'INTERNAL ERROR' in out['status']:
        st.error(f'Vyskytla sa interní chyba: {out["status"]}', icon="🚨")
    else:
        if out['status'] == 'OOPP':
            st.info('Predikce mimo Prahu muze byt nespolehliva', icon="ℹ️")
            # st.write(f'Predikce mimo Prahu muze byt nespolehliva')

        model_path = 'models/fitted_gp_low'
        gp_model = get_gp(model_path)

        X = out['data'][['long', 'lat']].to_numpy()
        mean_price, std_price = gp_model.predict(X, return_std=True)
        # price_gp = (mean_price * out['data']["usable_area"].to_numpy()).item()
        # std = (std_price * out['data']["usable_area"].to_numpy()).item()

        st.success('Predikce ceny Vaší nemovitosti :house: probehla úspešně')

        # OTHER MODELS
        model = Model(data=out['data'], inference=True, tune=False)
        pred_lower, pred_mean, pred_upper, shapy = model()

        locale.setlocale(locale.LC_ALL, '')
        pred_cena = " ".join("{0:n}".format(round(pred_mean.item())).split(','))
        st.subheader(f'🌲 Predikovaná cena Vašeho bytu je: {pred_cena}Kč.')

        render_bar_prediction(round(pred_lower.item()), round(pred_mean.item()), round(pred_upper.item()))


        price_per_m2_xgb = (pred_mean / out['data']['usable_area'].to_numpy()).item()
        gp_price = " ".join("{0:n}".format(round(mean_price.item())).split(','))
        gp_delta = " ".join(
            "{0:n}".format(round(price_per_m2_xgb - mean_price.item())).split(
                ','))

        with st.expander('Efekty priznaku na cenu bytu'):
            # fig = shap.plots.waterfall(shapy, show=False)
            # st_shap(shap.plots.waterfall(shapy), height=1000, width=1300)

            st.info('Pro zobrazeni nazvu priznaku prilozte k prislusnemu slopci')


            render_bar_plot_v2()

        # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
        with st.expander('                                                  Informace o okolí Vaší nemovitosti 🏠'
                         ''):

            col1, col2 = st.columns(2)

            with col1:
                air_quality = (6 - float(out["quality_data"]["air_quality"].item() if
                                         out["quality_data"]["air_quality"].item() != 'unknown' else 6)) * 20
                built_quality = (float(out["quality_data"]["built_density"].item() if
                                       out["quality_data"]["built_density"].item() != 'unknown' else 0)) * 20
                sun_quality = (6 - float(out["quality_data"]["sun_glare"].item() if
                                         out["quality_data"]["sun_glare"].item() != 'unknown' else 6)) * 20

                render_ring_gauge_quality(sun_quality,
                                          air_quality,
                                          built_quality)

            with col2:
                render_donut_plot()

            # st.write(f':sun_with_face: Slunečnost: {out["quality_data"]["sun_glare"].item()}')
            st.subheader(f':musical_note: Hlučnost v okoli: '
                     f'{str(out["quality_data"]["daily_noise"].item()) + "dB" if out["quality_data"]["daily_noise"].item() != 0 else "Data nedostupna"}')

            image = Image.open('../data/misc/hluk.png')
            st.image(image)

            price_advertised = None

            if not handmade and not out['data']['price_m2'].isna().any():
                price_advertised = " ".join("{0:n}".format(round(out['data']['price'].item())).split(','))
                price_xgb_delta = " ".join("{0:n}".format(round(pred_mean.item() -
                                                                out['data']['price'].to_numpy().item())).split(
                    ','))

        with st.expander('Rozdíly v cene'):
            if price_advertised is not None:
                col1, col2 = st.columns(2)
                col1.metric("Průměrná cena bytu v okolí", f"{gp_price} Kč/m2", f"{gp_delta} Kč/m2",
                            help="Indikator zobrazuje prumernu cenu bytov za m2 v dane oblasti. \n"
                                 "Nize je zobrazeny rozdil nase predikce oproti prumerne cene. \n"
                                 "Zelena znamena ze nase predikce udava cenu vyssi, cervena naopak znamena ze \n "
                                 "nase predikce zobrazuje nizsi cenu.")
                col2.metric("Navrhovana cena bytu", f"{price_advertised} Kč", f"{price_xgb_delta} Kč",
                            help=f"Indikator zobrazuje navrhovanou cenu bytu z uvedene url \n  {url}. \n"
                                 "Nize je zobrazeny rozdil nase predikce oproti navrhovane cene. \n"
                                 "Zelena znamena ze nase predikce udava cenu vyssi, cervena naopak znamena ze \n "
                                 "nase predikce zobrazuje nizsi cenu."
                            )
            else:
                col = st.columns(1)
                col[0].metric("Průměrná cena bytu v okolí", f"{gp_price} Kč/m2", f"{gp_delta} Kč/m2",
                              help="Indikator zobrazuje prumernu cenu bytov za m2 v dane oblasti. \n"
                                   "Nize je zobrazeny rozdil nase predikce oproti prumerne cene \n"
                                   "Zelena znamena ze nase predikce udava cenu vyssi, cervena naopak znamena ze \n "
                                 "nase predikce zobrazuje nizsi cenu."
                              )


selected = streamlit_menu(example=EXAMPLE_NO)

############## 1. stránka ##############
if selected == "Domů":
    st.header(f"Real e-state")
    st.markdown(
        ":sparkles: Naše vize je pomoci lidem predikovat ceny nemovitostí (bytů v Praze). Predikovat lze pomocí:")
    st.markdown("       - zadaného URL z sreality.cz nebo bezrealitky.cz,")
    st.markdown("       - pomocí ručně zadaných vlastností bytu.")
    st.markdown(
        ":sparkles: Dále můžeme investorům pomoci detekovat, jaké nemovitosti na trhu jsou podceněné nebo nadceněné a do kterých je lepší investovat.")
    st.markdown(
        ":sparkles: Bonusem bude dodání dalších informací o nemovitosti jako například hlučnost, obydlenost apod.")

############## 2. stránka ##############
if selected == "Predikce pomocí URL":
    st.header("Predikce ceny nemovitosti pomocí URL")
    # url
    url = st.text_input('URL nemovitosti (bytu v Praze) z sreality.cz or bezrealitky.cz')
    if url is None:
        pass
    else:
        url = str(url)
        if ('bezrealitky' or 'sreality') and 'praha' in url:
            with open('../data/predict_links.txt', 'w') as f:
                f.write(url)

    ############## MODELS ##############
    result_url = st.button('Predikuj!')
    if result_url:
        prediction(handmade=False, url=url)

if selected == "Predikce pomocí ručně zadaných příznaků":
    st.header(f"Predikce pomocí ručně zadaných příznaků")

    with st.expander("Zadej příznaky"):

        out = get_csv_handmade()

        ############## MODELS ##############
        result = st.button('Predikuj!')

    if result:
        print(type(out), out)
        df = pd.DataFrame(data={k: [v] for k, v in out.items()})
        df.to_csv('../data/predict_breality_scraped.csv', index=False)

        prediction(handmade=True)

############## 3. stránka ##############
if selected == "Kontakt":
    st.header(f"Kontakt")
    st.markdown(":copyright: Zkoukněte náš [GitHub](https://github.com/Many98/real_estate) :sunglasses:")


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


# Prague Apartment Price Prediction

This repository is dedicated to the development and application of machine learning techniques for predicting the prices of apartments in Prague.

## Requirements

-----------------------------------------------------------------------------------
Ensure you set up a Python environment before running the code. You can use one of the following commands:

1. ``conda env create -f environment.yml``
2. ``conda env create re && conda activate re && pip install -r requirements.txt``


Python Version: The codebase uses Python 3.10.6.


## Running the code

---------------------------------------------------------------------

#### 1. Command Line Interface (CLI)

For detailed instructions, use: `python main.py --help`

##### Examples:

Hyperparameter Search: `python main.py --train --tune` (data loaded from `../data/dataset.csv`)

Default Prediction: `python main.py` (runs prediction on data from ../data/dataset.csv and saves results in `../data/result.csv`)

Training with New Data: `python main.py --train --scrape` (scrapes new data and performs training)

#### 2. Web Interface
Run a local web server: `streamlit run web.py`


## Processing logic

-------------------------------------------------------------------------------------
The processing runs in two phases:

1. Training Phase: Crawlers obtain all advertisements.
2. Inference Phase: Users provide advertisement URL/data via the web app.

### Implementation Details
`ETL` Class: Handles data acquisition and preprocessing.

`Model` Class: Operates on data preprocessed by `ETL`. 

The model is based on XGBoost.
   
### Graphical proposal of processing logic 


![etl](https://user-images.githubusercontent.com/65658910/201643260-06bb1a57-564a-4413-9df0-c344095bff66.png)


## Web interface look
![home](https://github.com/Many98/real_estate/assets/65658910/da1915a7-f243-46eb-bbfb-d7e420defc22)


### Prediction using manually entered attributes
![by_hand00](https://github.com/Many98/real_estate/assets/65658910/a2219cd6-dd7e-41ba-95ea-64fb4c4d105e)
![by_hand](https://github.com/Many98/real_estate/assets/65658910/f82d2de8-f214-40c6-9c2e-26223aaaa600)

![prediction3](https://github.com/Many98/real_estate/assets/65658910/ba7ec5f1-2bf7-41b7-a34f-9c7f89bfd3bd)



### Prediction using url of sreality advertisement
![sreality](https://github.com/Many98/real_estate/assets/65658910/36d6c686-058b-48ee-bf76-e683167b9a76)

![prediction_by_url](https://github.com/Many98/real_estate/assets/65658910/9cea268c-e6e2-4cfe-b429-21bea2da548a)


### Effects of attributes on final price prediction
![effects_by_url](https://github.com/Many98/real_estate/assets/65658910/e9eb67ee-ca82-4b10-9e0c-c823a6961fd6)

![effects_by_url2](https://github.com/Many98/real_estate/assets/65658910/58a0b5af-4d43-45ee-8010-6743a8053eb2)


### Additional information about neighbourhoods of apartments
![add_by_url](https://github.com/Many98/real_estate/assets/65658910/6c45abc8-633b-44dd-b792-a7e21c78e84b)

![dist3](https://github.com/Many98/real_estate/assets/65658910/dc388c9a-ff78-47c6-a97c-e4accdc55742)




## Team members

-------------------------------------------------------------------
* Hanka Nguyenová (Team leader) 
* Daniel Karlík
* Emanuel Frátrik
* (Adam Šumník)

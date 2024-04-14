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

![home](https://github.com/Many98/real_estate/assets/65658910/345c92c4-762c-4618-b52a-c6be790e32da)

### Prediction using manually entered attributes
![by_hand00](https://github.com/Many98/real_estate/assets/65658910/5dddd1c8-b59d-41c1-a728-713d2d7f5da0)
![by_hand](https://github.com/Many98/real_estate/assets/65658910/1e19e335-37ad-4194-9b8d-2d0797424144)

![prediction3](https://github.com/Many98/real_estate/assets/65658910/75128a6e-c515-4d17-8e76-53554d6be844)


### Prediction using url of sreality advertisement
![sreality](https://github.com/Many98/real_estate/assets/65658910/8b7e7ac6-b044-4d89-a8a8-cdd1b62a44f3)

![prediction_by_url](https://github.com/Many98/real_estate/assets/65658910/56293b34-3cb8-4f4e-bbee-8ea67aee3472)


### Effects of attributes on final price prediction

![effects_by_url](https://github.com/Many98/real_estate/assets/65658910/eb8674f3-9600-4b82-976e-152cae085f4f)

![effects_by_url2](https://github.com/Many98/real_estate/assets/65658910/a5e32c1b-bb88-4b0e-8be9-0f2ddf7dd051)

### Additional information about apartments

![add_by_url](https://github.com/Many98/real_estate/assets/65658910/8bb4f407-3a62-43e7-b210-3a6f3dcbcdb9)

![dist3](https://github.com/Many98/real_estate/assets/65658910/e101e116-7ca6-43e2-ad91-b321c875051e)




## Team members

-------------------------------------------------------------------
* Hanka Nguyenová (Team leader) 
* Daniel Karlík
* Emanuel Frátrik
* (Adam Šumník)

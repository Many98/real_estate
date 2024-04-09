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

TODO


## Team members

-------------------------------------------------------------------
* Hanka Nguyenová (Team leader) 
* Daniel Karlík
* Emanuel Frátrik
* (Adam Šumník)

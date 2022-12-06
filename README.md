# Prediction of prices of apartments in Prague

### Requirements

-----------------------------------------------------------------------------------
Make sure you first create (e.g. conda) environmnent with either of commands

- ``conda env create -f environment.yml``
- ``conda env create re && conda activate re && pip install -r requirements.txt``

We use ``python 3.10.6``

### Run the code

---------------------------------------------------------------------
There are two options:

- use CLI (for details use `python main.py --help`) e.g.
  * `python main.py --train --tune` -> performs hyperparameter search (data loaded from `../data/dataset.csv`)
  * `python main.py` -> runs prediction on data from `../data/dataset.csv` and saves result in `../data/result.csv`
  * `python main.py --train --scrape` -> scrapes new data and performs training
- use web interface which can be launched locally using following command
  * ``streamlit run web.py``
  
There is no jupyter notebook demo as we (or just Many98) do not like Jupyter notebooks :sunglasses:


### Processing logic is as follows

-------------------------------------------------------------------------------------
Processing runs in two phases:
* `train` -> all advertisements are obtained by crawlers
* `inference` -> advertisement url/data is/are provided by user via web app

Implementation is also implemented in 2 main classes:
* `ETL` class handles data obtaining and preprocessings
* `Model` class operates on preprocessed data by `ETL`
   and the model itself is based on XGBoost
   
#### Graphical proposal of processing logic 


![etl](https://user-images.githubusercontent.com/65658910/201643260-06bb1a57-564a-4413-9df0-c344095bff66.png)


### Team members

-------------------------------------------------------------------
* Hanka Nguyenová (Team leader) 
* Daniel Karlík
* Emanuel Frátrik
* (Adam Šumník)

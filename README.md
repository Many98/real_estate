# Prediction of prices of apartments in Prague

### Team members

* Hanka Nguyenová (Team leader) 
* Adam Šumník
* Emanuel Frátrik

### Requirements

``python 3.10.6 is used``

just use ``conda env create -f environment.yml`` to create conda env 


### Processing logic is as follows
Processing runs in two phases:
* `train` -> all advertisements are obtained by crawlers
* `inference` -> advertisement url is provided by user via web app

Implementation is also implemented in 2 main classes:
* `ETL` class handles data obtaining and preprocessings
* `Model` class operates on preprocessed data by `ETL`

#### Graphical proposal of processing logic 


![etl](https://user-images.githubusercontent.com/65658910/201643260-06bb1a57-564a-4413-9df0-c344095bff66.png)

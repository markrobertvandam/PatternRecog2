# PatternRecog2 

## Requirements

To install all the requirements for both tasks run:
```
git clone https://github.com/markrobertvandam/PatternRecog2
pip3 install -r requirements.txt
```

## Data

To set-up the data either 
* Save the ready-made zip-file containing the git and all data for both Tasks

OR

* Save the provided BigCats and cats_projekat (https://www.kaggle.com/enisahovi/cats-projekat-4/version/1) folders in 
Task1/data/ and run data_filter.py which will create cats_projekat_filtered automatically
* Save the provided Genes data in Task1/data/
* Download `creditcard.csv` in the Task2/data/ folder


## Running Task 1

To run Task 1 of the assigment you can simply run the main.py inside the Task1 folder with a command:
* for image classification: ```python3 main.py cats command```
* for genes/number classification: ```python3 main.py genes command```

The command options for both pipelines are as follows (will use cats as example):
* ```python3 main.py cats tune``` # To run small grid-search to find best params
* ```python3 main.py cats tune``` # To run big grid-search to find best params (not recommended)
* ```python3 main.py cats test``` # Test run with 80% training and 20% test 
* ```python3 main.py cats cross-val``` # Basically 5 test-runs with different splits of 80% training and 20% test
* ```python3 main.py cats cluster``` # Runs the clustering algorithm and outputs clustering scores
* ```python3 main.py cats full-run``` # command to run test, cross-val and ensemble all at once. (Saves time because data and 
feature selection/extraction only has to be done once)

The following 2 commands only work for cats
* ```python3 main.py cats augment```  # cross-val run with augmented images
* ```python3 main.py cats ensemble``` # cross-val runs with all possible ensemble combinations and their test performance.


## Running Task 2
* Run ```python3 main.py data results``` inside the Task2 folder

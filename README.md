# PatternRecog2 

## Requirements

To install all the requirements for both tasks run:
```
git clone https://github.com/markrobertvandam/PatternRecog2
pip3 install -r requirements.txt
```

## Data

To set-up the data either 
* Save the ready-made zip-file called cats_projekat_filtered in Task1/data/
* Save the provided BigCats and cats_projekat folders in Task1/data/ and run data_filter.py which will create cats_projekat_filtered automatically


## Running Task 1

To run Task 1 of the assigment you can simply run the main.py inside the Task1 folder with a command:
* for image classification: ```python3 main.py cats command```
* for genes/number classification: ```python3 main.py genes command```

The command options for both pipelines are as follows (will use cats as example):
* ```python3 main.py cats tune``` # To run grid-search to find best params
* ```python3 main.py cats test``` # Test run with 80% training and 20% test 
* ```python3 main.py cats cross-val``` # Basically 5 test-runs with different splits of 80% training and 20% test
* ```python3 main.py cats ensemble``` # Test-run followed by runs with all 4 possible ensemble combinations and their test performance.
* ```python3 main.py cats full-run``` # command to run test, cross-val and ensemble all at once. (Saves time because data and 
feature selection/extraction only has to be done once)

## Running Task 2
```python3 main.py``` inside the Task2 folder
# Version Control Collaboration Best Practice
* If you work on some directory that you know there would be other people working on it, use a feature branch from dev branch
* Make sure your feature branch is up-to-date with the dev branch before doing anything
* Make sure your working branch is up-to-date with corresponding remote branch
* ...


# Setup Local Dev Environment

## 1.virtualenv setup
The following example use python 3.10. The version of python must be greater than 3.6.
Make sure you run the following command in the backend directory so that the venv directory is in parallel with src directory.
```bash
pip3.10 install virtualenv
virtualenv venv --python=python3.10
```

## 2.dependencies installations
Install dependencies in virtualenv
```bash
source venv/bin/activate
pip install -r requirements.txt -e .
```

## 3.local startup
```bash
source venv/bin/activate
python src/run.py
```

## 4.swagger playground
The docs page gives access to API interface and allow us to directly test API through UI.
```bash
http://127.0.0.1:8000/docs
```

## 5.import data
When you first run the application, you should find empty database file called transations.sqlite3. 
We need to init the database by importing existing transactions data.
In order to do this, install the sqlite3 CLI and enter into the interactive shell.
Make sure you have new versions of sqlite3.

```bash
brew install sqlite3
sqlite3 transactions.sqlite3
```

In the sqlite3 interactive shell, type the following commands to import the csv file called dump.csv.
> IMPORTANT: For windows, don't include the --skip 1 option but for Mac the --skip 1 option is compulsory
```bash
.mode csv transactions
.separator ","
.import --csv --skip 1 transactions_dump.csv transactions
select count() from transactions;

.import --csv --skip 1 predictions_dump.csv predictions
select count() from predictions;
```
If count() gives 138701 rows then the import is successful.

> To reinit the database, just delete the transactions.sqlite3 file, restart the application and re-import the data.


# How to Debug
Please refer to this documentation: https://fastapi.tiangolo.com/tutorial/debugging/. 
Pycharm has a known issue with python 3.11 together with uvicorn: https://youtrack.jetbrains.com/issue/PY-57217

# How to Test
Testing files are stored in tests directory. Naming conventions are following the best practice of pytest.
Run the following command to test.
```bash
pytest
```


# How to generate dump files
#### 1. prediction_dump.csv
we use the [prediction_generator.py](../classifier/prediction_generator.py) to generate prediction probability based on given model (for now, random forest is used as it is the best based on our experiments)

#### 2. transactions_dump.csv
we use the [transactions_dump.ipynb](./transactions_dump.ipynb) to generate all the historic transactions given by adyen.
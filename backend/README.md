# Dev Setup

## 1.virtualenv setup
The following example use python 3.11. The version of python must be greater than 3.6.
```bash
pip3.11 install virtualenv
virtualenv venv --python=python3.11
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
```bash
.mode csv transactions
.separator ","
.import --csv --skip 1 dump.csv transactions
select count() from transactions;
```
If count() gives 138701 rows then the import is successful.

> To reinit the database, just delete the transactions.sqlite3 file, restart the application and re-import the data.

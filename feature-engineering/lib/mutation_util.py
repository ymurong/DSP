import datetime


def date(row):
    year = row[3]
    day_of_year = int(row[6])
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    return date

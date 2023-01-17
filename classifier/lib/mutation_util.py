import datetime
from calendar import monthrange


def date(row):
    year = row[3]
    day_of_year = int(row[6])
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    return date


def daysOfMonth(start_month, end_month, year=2021):
    """ calculate how many days between start_month and end_month"""
    days = 0
    for month in range(start_month, end_month + 1):
        days += monthrange(year, month)[1]
    return days

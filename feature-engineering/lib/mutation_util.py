import datetime
import pandas as pd


def date(row):
    year = row[3]
    day_of_year = int(row[6])
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    return date


def get_tx_datetime(transaction):
    year = transaction.year
    day_of_year = int(transaction.day_of_year)
    hour_of_day = transaction.hour_of_day
    minute_of_hour = transaction.minute_of_hour
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    tx_datetime = datetime.datetime.combine(date, datetime.time(hour_of_day, minute_of_hour))
    return tx_datetime


def is_weekend(tx_datetime):
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5

    return int(is_weekend)


def is_night(tx_datetime):
    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour <= 6

    return int(is_night)


def get_card_spending_behaviour_features(transactions, windows_size_in_days=[1, 7, 30]):
    # Let us first order transactions chronologically
    transactions = transactions.sort_values('tx_datetime')

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    transactions.index = transactions.tx_datetime

    # For each window size
    for window_size in windows_size_in_days:
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = transactions['eur_amount'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = transactions['eur_amount'].rolling(str(window_size) + 'd').count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        transactions['card_nb_tx_' + str(window_size) + 'day_window'] = list(NB_TX_WINDOW)
        transactions['card_avg_amount_' + str(window_size) + 'day_window'] = list(AVG_AMOUNT_TX_WINDOW)

    # Reindex according to transaction IDs
    transactions.index = transactions.psp_reference

    # And return the dataframe with the new features
    return transactions


def get_diff_tx_time(transactions):
    transactions = transactions.sort_values('tx_datetime')
    transactions["diff_tx_time_in_hours"] = transactions.tx_datetime.diff().dt.total_seconds() / 3600
    transactions["diff_tx_time_in_hours"].fillna(0, inplace=True)
    return transactions


def is_diff_previous(transactions, feature):
    transactions = transactions.sort_values('tx_datetime')
    transactions.index = transactions.psp_reference
    transactions["is_diff_previous_" + feature] = False
    # exclude na values when comparing
    transactions_notna = transactions.dropna(subset=[feature])
    transactions_notna = transactions_notna.sort_values('tx_datetime')
    transactions_notna.loc(axis=1)["is_diff_previous_" + feature] = (transactions_notna[feature] != transactions_notna[feature].shift(periods=1))
    transactions_notna = transactions_notna[["psp_reference", "is_diff_previous_" + feature]]
    transactions_notna["is_diff_previous_" + feature].iloc[0] = False
    transactions.update(transactions_notna)
    return transactions


def get_count_risk_rolling_window(transactions, feature, delay_period=7, windows_size_in_days=[1, 7, 30]):
    check_nan = transactions[feature].isnull().values.any()
    if check_nan:
        for window_size in windows_size_in_days:
            transactions[feature + '_nb_tx_' + str(window_size) + 'day_window'] = 0
            transactions[feature + '_risk_' + str(window_size) + 'day_window'] = 0
        transactions.index = transactions.psp_reference
        return transactions

    transactions = transactions.sort_values('tx_datetime')

    transactions.index = transactions.tx_datetime

    NB_FRAUD_DELAY = transactions['has_fraudulent_dispute'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = transactions['has_fraudulent_dispute'].rolling(str(delay_period) + 'd').count()

    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = transactions['has_fraudulent_dispute'].rolling(
            str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = transactions['has_fraudulent_dispute'].rolling(
            str(delay_period + window_size) + 'd').count()

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        transactions[feature + '_nb_tx_' + str(window_size) + 'day_window'] = list(NB_TX_WINDOW)
        transactions[feature + '_risk_' + str(window_size) + 'day_window'] = list(RISK_WINDOW)

        # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
        transactions[feature + '_risk_' + str(window_size) + 'day_window'].fillna(0, inplace=True)

    transactions.index = transactions.psp_reference
    return transactions

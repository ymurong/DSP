from sklearn.preprocessing import StandardScaler
import pandas as pd


def standardize(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled = scaler.fit_transform(df.to_numpy())
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df

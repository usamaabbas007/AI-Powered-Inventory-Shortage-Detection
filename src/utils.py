
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, label_col="shortage_event"):
    df = df.copy()
    df = df.dropna()
    X = df.drop(columns=[label_col])
    y = df[label_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler

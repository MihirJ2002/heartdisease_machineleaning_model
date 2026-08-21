# import data manipulation libraries
import pandas as pd
import numpy as np
filepath = "https://raw.githubusercontent.com/MihirJ2002/heartdisease_machineleaning_model/refs/heads/main/data/heart_disease_risk_2026.csv"


def data_loader():

    df = pd.read_csv(filepath)

    return df

    
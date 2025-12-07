import sys
import os
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path 
import pandas as pd

#from scripts.download_data import download_Data
#from scripts.correlation import correlation
#from scripts.prediction import prediction_model_train
from src.scripts.show_data import show_data

# guardamos los datos en la carpeta
# Load data
original_path = "src/data/datos_barcelona_fusionados.csv"

def main():

    # # Load data
    # data = download_Data()
    data = pd.read_csv(original_path)

    show_data(data)

    # # Data processing
    # correlation_Data = correlation(data)

    # # Prediction with RFR
    # prediction_model_train(correlation_Data)


if __name__ == "__main__":
    main()
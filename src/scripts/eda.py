import streamlit as st

from utils.data_analysis import basic_stats
from utils.visualize_data import visualization_Data

def eda_interactivo(data):
    st.header("🔍 Exploratory Data Analysis (EDA)")

    basic_stats(data)
    visualization_Data(data)
    
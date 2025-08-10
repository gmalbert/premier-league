import streamlit as st
import pandas as pd
from os import path

DATA_DIR = 'data_files/'

st.set_page_config(page_title="Premier League Historical Data", layout="wide")

st.image(path.join(DATA_DIR, 'Premier_league_logo.png'))
st.title("Premier League Historical Data Viewer")
st.write(
    """
    Welcome! This app displays the historical Premier League data.
    Upload or update your data file in the `data_files` folder as `historical_data.csv`.
    """
)


csv_path = path.join(DATA_DIR, 'combined_historical_data.csv')

if path.exists(csv_path):
    df = pd.read_csv(csv_path, sep='\t')
    st.subheader("Historical Data")
    st.dataframe(df)
else:
    st.warning(f"No historical data file found at `{csv_path}`. Please add your CSV file to get started.")
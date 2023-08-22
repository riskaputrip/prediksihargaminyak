import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from datetime import datetime
from PIL import Image
from tabulate import tabulate
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.title("DATA HARGA")
st.subheader("HARGA MINYAK GORENG KEMASAN BERMERK 1 (KG) DI KOTA BANDUNG")
st.markdown("---")


#Functions for each of the pages
### --- LOAD DATAFRAME
excel_file = 'Data_Minyak.xlsx'
sheet_name = 'Data_Minyak'

df = pd.read_excel(excel_file,
                   sheet_name=sheet_name,
                   usecols='A:B',
                   header=0)

df['Date'] = pd.to_datetime(df['Date']).dt.date

# Sidebar filter
st.sidebar.header("Please Filter Here:")

# Tanggal Mulai
start_date = st.sidebar.date_input('Tanggal Mulai', min_value=df['Date'].min(), max_value=df['Date'].max(), value=df['Date'].min())

# Tanggal Berakhir
end_date = st.sidebar.date_input('Tanggal Berakhir', min_value=start_date, max_value=df['Date'].max(), value=df['Date'].max())

# Filter data berdasarkan tanggal
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Menampilkan tabel yang lebih bagus menggunakan tabulate
table = tabulate(filtered_df, headers='keys', tablefmt='pipe')
st.markdown(table, unsafe_allow_html=True)
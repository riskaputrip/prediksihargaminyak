### --- LOAD DATAFRAME
    excel_file = 'Data_Minyak_2019.xlsx'
    sheet_name = 'DATA_MINYAK_2019'

    df = pd.read_excel(excel_file,
                   sheet_name=sheet_name,
                   usecols='A:B',
                   header=0)

    st.dataframe(df)

    # Functions for each of the pages
    def data_minyak2020(df):
        ### --- LOAD DATAFRAME
        excel_file = 'Data_Minyak_2019.xlsx'
        sheet_name = 'DATA_MINYAK_2020'

    df = pd.read_excel(excel_file,
                   sheet_name=sheet_name,
                   usecols='A:B',
                   header=0)

    st.dataframe(df)

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
start_date = st.date_input('Start date', today)
end_date = st.date_input('End date', tomorrow)
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')


st.sidebar.header("Please Filter Here:")
date = st.sidebar.multiselect("Pilih tahun dataset",
                                   options=df["Date"].unique(),
                                   default=df["Date"].unique()
                                   )

#yt
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('lstm_model.pkl','wb'))

df = pd.read_excel("Data_Minyak.xlsx")
df['Date'] = pd.to_datetime(df['Date'], format='%Y')
df.set_index(['Date'], inplace=True)

st.title('Prediksi Harga Minyak Goreng Kemasan Bermerk 1 (KG) di Bandung')
date = st.slider("Tentukan Hari",1,100, step=1)

pred = model.regressor(date)
pred = pd.DataFrame(pred,columns=['Price'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        st.pyplot(fig)

df_selection = df.query(
        "Date == @date"
    )

d3 = st.date_input("range, no dates", [])
st.write(d3)

d3 = st.date_input("Range, one date", [datetime.date(2019, 7, 6)].unique())
st.write(d3)

d5 = st.date_input("date range without default", [datetime.date(2019, 7, 6), datetime.date(2019, 7, 8)].unique())
st.write(d5)

st.write(df_selection.head())

#contoh
import numpy as np
import pandas as pd
import pickle 
import streamlit as st

def load_model():
    with open("lstm_model.pkl", "rb") as pickle_file:
        model = pickle.load(pickle_file)
    return model

def predict_price(model, date):
    prediction = model.predict([[date]]) # Melakukan prediksi menggunakan model
    return prediction

def main():
    st.title("Admission prediction APP using ML") # Judul aplikasi
    html_temp = """
        <div>
        <h2>Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True) # Menggunakan HTML untuk tampilan

    # Memuat model
    model = load_model()

    # Input tanggal
    date = st.text_input("Date")

    result = ""
    if st.button("Predict"):
        result = predict_price(model, date)
    st.success("Prediction: {}".format(result))

if __name__ == '__main__':
    main()

def load_model():
    with open("lstm_model.pkl", "rb") as pickle_file:
        model = pickle.load(pickle_file)
    return model


#prediksi harga code
import numpy as np
import pandas as pd 
import tensorflow as tf
import pickle 
import keras
import streamlit as st


def load_model():
    model = tf.keras.saving.load_model("model.keras")

def predict_price(model, date):
    prediction = model.predict([[date]]) # Melakukan prediksi menggunakan model
    return prediction
 
def main():
    st.title("Prediksi Harga Minyak Goreng Kemasan Bermerk 1 (KG)") # Judul aplikasi
    html_temp = """
        <div>
        <h2>Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True) # Menggunakan HTML untuk tampilan

    #Streamlit Selection
date = df['Date'].unique().tolist()

date_selection = st.slider('Date:',
                           min_value= min(date),
                           max_value= max(date),
                           value=(min(date),max(date)))

    # Memuat model
model = load_model()

    # Input tanggal
date = st.text_input("Date")

result = ""
if st.button("Predict"):
        result = predict_price(model, date)

st.success("Prediction: {}".format(result))

if __name__ == '__main__':
    main()


# Prediksi harga 2
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle 
import keras
import matplotlib.pyplot as plt
import streamlit as st

#Convert date coulmns to specific format
dateparse = lambda x: pd.to_datetime(x, format='%d/%m/%Y')

df = pd.read_csv("Data_Minyak.csv", sep =";", parse_dates=['Date'], date_parser=dateparse)

def load_model():
    model = tf.keras.saving.load_model("model.keras")

def predict_price(model, date):
    prediction = model.predict([[date]]) 
    # Melakukan prediksi menggunakan model
    return prediction

def main():
    st.title("Prediksi Harga Minyak Goreng Kemasan Bermerk 1 (KG)") # Judul aplikasi
    html_temp = """
        <div>
        <h2>Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True) # Menggunakan HTML untuk tampilan

#Streamlit Selection
date = st.slider("Tentukan Hari",1,100, step=1)

predict_price = model.regressor(date)
predict_price = pd.DataFrame(model,columns=['Price'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(model)
    with col2:
        fig, ax = plt.subplots()
        st.pyplot(fig)


    # Memuat model
model = load_model()

    # Input tanggal
date = st.text_input("Date")

result = ""
if st.button("Predict"):
        result = predict_price(model, date)
st.success("Prediction: {}".format(result))

if __name__ == '__main__':
    main()


#
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle 
import keras
import matplotlib.pyplot as plt
import streamlit as st

#Convert date coulmns to specific format
dateparse = lambda x: pd.to_datetime(x, format='%d/%m/%Y')

df = pd.read_csv("Data_Minyak.csv", sep =";", parse_dates=['Date'], date_parser=dateparse)

def load_model():
    model = tf.keras.saving.load_model("model.keras")
    return model

def predict_price(model, date):
    #Model
    prediction = model.predict([[date]]) # Melakukan prediksi menggunakan model
    return prediction

def main():
    st.title("Prediksi Harga Minyak Goreng Kemasan Bermerk 1 (Kg)") # Judul aplikasi
    html_temp = """
        <div>
        <h2>Admission Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True) # Menggunakan HTML untuk tampilan

#Streamlit Selection
date = st.slider("Tentukan Hari",1,100, step=1)

predict_price = load_model.regressor(date)
predict_price = pd.DataFrame(load_model,columns=['Price'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(load_model)
    with col2:
        fig, ax = plt.subplots()
        st.pyplot(fig)


    # Memuat model
model = load_model()

    # Input tanggal
date = st.text_input("Date")

result = ""
if st.button("Predict"):
        result = predict_price(model, date)
st.success("Prediction: {}".format(result))

if __name__ == '__main__':
    main()
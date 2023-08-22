import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
import keras
import math 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import streamlit as st
import seaborn as sns


#Convert date coulmns to specific format
#dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
#df["tanggal"] = pd.to_datetime(df["tanggal"], format= '%d/%m/%Y')
#df.set_index('tanggal', inplace=True)
#df=df.loc[datetime.date(year=2020,month=1,day=1):]
#df = pd.read_csv("Data_Minyak.csv", sep =";", parse_dates=['Date'], date_parser=dateparse)


df = pd.read_csv("Data_Minyak.csv", sep =";")
df["Date"] = pd.to_datetime(df["Date"], format= '%d/%m/%Y')
#Sort dataset by column Date
df = df.sort_values('Date')
df = df.groupby('Date')['Price'].sum().reset_index()
df.set_index('Date', inplace=True)
df=df.loc[datetime.date(year=2019,month=1,day=2):]

#Read dataframe info
def DfInfo(df_initial):
    # gives some infos on columns types and numer of null values
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                               rename(index={0: 'null values (%)'}))
    return tab_info

y = df['Price'].resample('MS').mean()

# normalize the data_set
sc = MinMaxScaler(feature_range = (0, 1))
df = sc.fit_transform(df)

# split into train and test sets
train_size = int(len(df) * 0.80)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]

# convert an array of values into a data_set matrix def
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)

# reshape into X=t and Y=t+1
look_back =4
X_train,Y_train,X_test,Ytest = [],[],[],[]
X_train,Y_train=create_data_set(train,look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test,Y_test=create_data_set(test,look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = tf.keras.saving.load_model("model.keras")

#print('3 nilai RSME pada 3 kali training model:')

#Pelatihan model LSTM
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# invert predictions
train_predict1 = sc.inverse_transform(train_predict)
Y_train1 = sc.inverse_transform([Y_train])
test_predict1 = sc.inverse_transform(test_predict)
Y_test1 = sc.inverse_transform([Y_test])

trainScore = metrics.mean_squared_error(Y_train1[0], train_predict1[:,0]) ** .5
#print('Train Score: %.2f RMSE' % (trainScore))
testScore = metrics.mean_squared_error(Y_test1[0], test_predict1[:,0]) ** .5
#print('Test Score: %.2f RMSE' % (testScore))

#Compare Actual vs. Prediction
aa=[x for x in range(180)]
plt.figure(figsize=(15,6))
plt.plot(Y_test1[0], 'b', label="actual")
plt.plot(test_predict1, 'y', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time Step', size=15)
plt.legend(fontsize=15)
plt.show()

# Compare train data Actual vs. Prediction
plt.style.context("seaborn-white")
plt.figure(figsize=(15,6))
plt.plot(Y_train1[0], 'b', label="actual")
plt.plot(train_predict1, 'y', label="prediction")
plt.tight_layout()
plt.title('Train Data')
# sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.legend(fontsize=15)
plt.show()

# Compare test data Actual vs. Prediction
plt.figure(figsize=(15,6))
plt.plot(Y_test1[0], 'b', label="actual")
plt.plot(test_predict1, 'y', label="prediction")
plt.tight_layout()
# sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=11)
plt.xlabel('Total Test Data', size=11)
plt.legend(fontsize=15)
plt.show()

train_predict2 = train_predict1.T
train_predict2 = list(train_predict2[0])
#print(train_predict2)

test_predict2 = test_predict1.T
test_predict2 = list(test_predict2[0])
#print(test_predict2)

#Sort dataset by column Date
df = pd.read_csv("Data_Minyak.csv", sep =";")
df["Date"] = pd.to_datetime(df["Date"], format= '%d/%m/%Y')
df = df.sort_values('Date')
df = df.groupby('Date')['Price'].sum().reset_index()
df.set_index('Date', inplace=True)
df=df.loc[datetime.date(year=2019,month=1,day=2):]

# tabel untuk data train beserta hasil prediksinya
df_train = df[:len(train_predict2)]
df_train['Predictions'] = train_predict2

# tabel untuk data test beserta hasil prediksinya
df_test = df[-(len(test_predict2)):]
df_test['Predictions'] = test_predict2

# Menggunakan Streamlit untuk menampilkan slider dan grafik prediksi
st.title('Prediksi Harga Minyak Goreng Kemasan Bermerk 1(Kg)')

# Create a sidebar with title and description
st.sidebar.title("Price Prediction App")
st.sidebar.markdown("Aplikasi ini memprediksi harga minyak goreng kemasan bermerk 1 (Kg) selama 100 hari kedepan")
st.sidebar.markdown("Pilih jumlah hari untuk menampilkan prediksi menggunakan penggeser di samping ini:")

#Prediksi 100 Hari
len(test)
#df["Date"] = pd.to_datetime(df["Date"], format= '%d/%m/%Y')
#df.set_index('Date', inplace=True)
#df=df.loc[datetime.date(year=2019,month=1,day=2):]
date = np.array('2023-03-02', dtype=np.datetime64)
#date
#date = date + np.arange(100)
x_input = test[105:].reshape(1,-1)
#x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
pred_100=[]
n_steps= 100
i=0
while(i<100):

    if len(temp_input)>=100:
      x_input=np.array(temp_input[-100:])
      print("{} day input {}".format(i,x_input))
      x_input=x_input.reshape(1,-1)
      x_input = x_input.reshape((1, 100, 1))

      pred_new = model.predict(x_input)
      print("{} day output {}".format(i,pred_new))
      temp_input.extend((pred_new[0]/10).tolist())
      temp_input=temp_input[1:]

      pred_100.extend(pred_new.tolist())
      i=i+1
    else:
      x_input = np.array(temp_input, dtype=np.float32)
      x_input= x_input.reshape(1, len(temp_input), 1)
      x_input = x_input.reshape((1, n_steps,1))
      
      pred_new = model.predict(x_input)
      print(pred_new[0])
      temp_input.extend(pred_new[0].tolist())
      print(len(temp_input))
      pred_100.extend(pred_new.tolist())
      i=i+1

pred_100 = np.array(pred_100)/10
pred_100_inv = sc.inverse_transform(pred_100)
df_100 = pd.DataFrame(pred_100_inv, columns=['prediction 100'])
# Generate a list of dates for the next 100 days starting from March 2, 2023
start_date = pd.to_datetime('2023-03-02')
date = pd.date_range(start_date, periods=100, freq='D')
df_100 = pd.DataFrame(pred_100_inv, columns=['prediction 100'])
df_100['Date'] = date  # Menambahkan kolom 'Date' dengan nilai date
df_100.set_index('Date', inplace=True)  # Mengatur 'Date' sebagai indeks
#df_100

day_test = np.arange(1, len(df_test['Price'])+1)
day_100 = np.arange(len(df_test['Price'])+1, len(df_test['Price'])+101)

plt.figure(figsize=(15,6))
plt.plot(pred_100_inv, 'y')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Date', size=15)
plt.legend(fontsize=15)
plt.show()

plt.figure(figsize=(15,6))
plt.plot(day_test, np.array(df_test['Price']), label="actual")
plt.plot(day_100, pred_100_inv.T[0], label="predictions")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time Step', size=15)
plt.legend(fontsize=15)
plt.show()

#Streamlit Selection

# Mendefinisikan tanggal untuk prediksi
start_date = pd.to_datetime('2023-03-02')
date = pd.date_range(start_date, periods=100, freq='D')

df = pd.read_csv("Data_Minyak.csv", sep =";")
df["Date"] = pd.to_datetime(df["Date"], format= '%d/%m/%Y')

def create_price_prediction_chart(pred_100_inv, day_range):
    # Assuming pred_100_inv is a numpy array
    pred_100_inv_df = pd.DataFrame({'Date': df_100.index, 'prediction 100': pred_100_inv[:, 0]})
    
    # Plot hasil prediksi
    plt.figure(figsize=(15, 6))
    plt.plot(df['Date'], df['Price'], label='Aktual')
    plt.plot(pred_100_inv_df['Date'][:day_range], pred_100_inv_df['prediction 100'][:day_range], label='Prediksi')
    plt.title('Prediksi Harga')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(plt)  # Display the plot using st.pyplot

# Streamlit Selection
day_range = st.slider("Tentukan Hari :", min_value=1, max_value=len(day_100), value=100, step=1)

if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(df_100.head(day_range))
    with col2:
        # Buat grafik prediksi harga menggunakan fungsi yang telah dibuat
        create_price_prediction_chart(pred_100_inv, day_range)

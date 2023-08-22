import streamlit as st
from PIL import Image
import numpy as np

st.title("LONG SHORT TERM MEMORY")

tab1, tab2 = st.tabs(["🤷‍♀️ Apa itu LSTM?", "👩‍💻Persiapan Data"])
data = np.random.randn(10, 1)

tab1.subheader("Apa itu Long Short Term Memory?")
tab1.markdown("Long Short Term Memory adalah arsitektur RNN yang dimodifikasi yang menangani masalah menghilangnya dan meledaknya gradien dan mengatasi masalah pelatihan pada urutan panjang dan mempertahankan memori.")
tab1.subheader("Struktur Long Short Term Memory")
tab1.markdown("1. INPUT GATE")
tab1.markdown("Input Gate memutuskan informasi baru apa yang akan disimpan dalam memori jangka panjang . Ini hanya bekerja dengan informasi dari input saat ini dan memori jangka pendek dari langkah waktu sebelumnya. Oleh karena itu, harus menyaring informasi dari variabel-variabel tersebut yang tidak berguna.")
image = Image.open('inputgate.jpg')
tab1.image(image, caption='Input Gate')

tab1.markdown("Secara matematis, hal ini dicapai melalui penerapan dua lapisan. Lapisan pertama berfungsi sebagai filter yang menentukan informasi mana yang diizinkan untuk berlalu dan informasi mana yang akan diabaikan. Untuk membuat lapisan ini, langkah pertama adalah menggabungkan memori jangka pendek dari iterasi sebelumnya dengan masukan saat ini. Gabungan ini kemudian disalurkan melalui fungsi sigmoid. Fungsi sigmoid mengubah nilai menjadi rentang antara 0 dan 1. Nilai 0 mengindikasikan bahwa bagian informasi tersebut dianggap tidak signifikan, sementara nilai 1 menunjukkan bahwa informasi tersebut akan diperhitungkan. Fase ini membantu dalam menentukan nilai-nilai yang perlu dipertahankan dan digunakan, serta nilai-nilai yang bisa diabaikan. Selama proses pelatihan melalui propagasi balik (backpropagation), bobot di fungsi sigmoid diperbarui sedemikian rupa sehingga lapisan ini belajar untuk hanya mengizinkan informasi yang bermanfaat untuk berlalu, sambil mengabaikan fitur yang dianggap kurang relevan.")
tab1.markdown("Lapisan kedua mengambil memori jangka pendek dan input saat ini juga dan meneruskannya melalui fungsi aktivasi, biasanya fungsi $$tanh $$ , untuk mengatur jaringan.")
tab1.markdown("Keluaran dari 2 lapisan ini kemudian dikalikan, dan hasil akhirnya mewakili informasi yang akan disimpan dalam memori jangka panjang dan digunakan sebagai keluaran.")

tab1.markdown("2. FORGET GATE")
tab1.markdown("Forget bertugas untuk menentukan informasi yang harus diperoleh atau diabaikan dari memori jangka panjang. Proses ini dilakukan dengan mengalikan memori jangka panjang yang diinputkan dengan forget vector yang dihasilkan dari input saat ini dan juga memori jangka pendek yang masuk.")
image = Image.open('forgetgate.jpg')
tab1.image(image, caption='Forget Gate')
tab1.markdown("Seperti halnya lapisan pertama pada Input Gate, Forget vector juga berfungsi sebagai lapisan filter selektif. Proses untuk mendapatkan forget vector melibatkan melewati memori jangka pendek dan input saat ini melalui fungsi sigmoid, serupa dengan langkah pada lapisan pertama di Input gate sebelumnya. Namun, pada langkah ini, digunakan bobot yang berbeda. Forget vector akan terdiri dari nilai-nilai 0 dan 1, yang akan dikalikan dengan memori jangka panjang. Hasil perkalian ini akan menentukan bagian mana dari memori jangka panjang yang akan dijaga.")
tab1.markdown("Keluaran dari Input Gate dan Forget Gate akan mengalami penambahan yang tepat untuk memberikan versi baru memori jangka panjang , yang akan diteruskan ke sel berikutnya. Memori jangka panjang baru ini juga akan digunakan di gerbang terakhir, Output Gate .")
tab1.markdown("3. OUTPUT GATE")
tab1.markdown("Output Gate akan menggunakan input saat ini, memori jangka pendek sebelumnya, dan memori jangka panjang yang baru saja dihitung untuk menghasilkan versi baru dari memori jangka pendek atau status tersembunyi. Versi baru ini akan diteruskan ke sel pada langkah waktu berikutnya. Selain itu, keluaran dari langkah waktu saat ini juga bisa diperoleh dari status tersembunyi ini.")
image = Image.open('outputgate.jpg')
tab1.image(image, caption='Output Gate')
tab1.markdown("Memori jangka pendek dan jangka panjang yang dihasilkan oleh gerbang ini kemudian akan dibawa ke sel berikutnya untuk diulangi prosesnya. Keluaran dari setiap langkah waktu dapat diperoleh dari memori jangka pendek, juga dikenal sebagai hidden state.")


tab2.subheader("PERSIAPAN DATA UNTUK MODEL LSTM")
tab2.subheader("Konversi Format Tanggal")
tab2.markdown("Data tanggal dalam format string ('dd/mm/yyyy') diubah menjadi objek tanggal yang sesuai menggunakan fungsi pd.datetime.strptime(). Ini memastikan bahwa data tanggal terbaca dengan benar oleh sistem.")
tab2.subheader("Membaca dan Memproses Data")
tab2.markdown("- Data dibaca dari file CSV ('Data_Minyak.csv') menggunakan pd.read_csv().")
tab2.markdown("- Data kemudian diurutkan berdasarkan kolom tanggal ('Date') menggunakan sort_values().")
tab2.markdown("- Dataset dikelompokkan berdasarkan tanggal dan harga dijumlahkan untuk setiap tanggal menggunakan groupby()")
tab2.subheader("Menghitung Statistik Data")
tab2.markdown("Fungsi DfInfo() dibuat untuk memberikan informasi tentang dataset. Ini termasuk jenis kolom, jumlah nilai null, dan persentase nilai null dalam dataset.")
tab2.subheader("Resampling Data")
tab2.markdown("- Data harga diresampling ke frekuensi bulanan (MS) menggunakan metode .resample('MS').")
tab2.markdown("- Rata-rata harga dihitung untuk setiap bulan dengan menggunakan .mean().")
tab2.subheader("Normalisasi Data")
tab2.markdown("MinMaxScaler digunakan untuk melakukan normalisasi data harga ke dalam rentang 0 hingga 1. Hal ini penting untuk melatih model neural network seperti LSTM, karena skala data yang seragam membantu dalam konvergensi pelatihan yang lebih baik.")
tab2.subheader("Pembagian Data Train dan Test")
tab2.markdown("Data dibagi menjadi data latih (train) dan data uji (test) dengan proporsi 80:20. Ini dilakukan dengan menghitung jumlah data yang akan digunakan sebagai data latih dan sisanya sebagai data uji.")
tab2.subheader("Fungsi Pembentukan Dataset")
tab2.markdown("- Fungsi create_data_set() dibuat untuk membentuk pasangan data X (input) dan Y (output) yang sesuai untuk pelatihan dan pengujian model.")
tab2.markdown("- Fungsi ini mengambil sejumlah data sebelumnya sebanyak look_back (dalam hal ini, 4) sebagai input dan menghasilkan data berikutnya sebagai output.")
tab2.subheader("Pemformatan Data Untuk Model LSTM")
tab2.markdown("- Data latih dan data uji diubah menjadi format yang sesuai untuk masukan model LSTM dengan menggunakan np.reshape().")
tab2.markdown("- Reshaping ini penting karena model LSTM menerima masukan dalam bentuk tiga dimensi: jumlah sampel, jumlah time steps, dan jumlah fitur.")



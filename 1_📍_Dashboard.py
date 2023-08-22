import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("SELAMAT DATANG!")
st.subheader("DI DASHBOARD PREDIKSI HARGA MINYAK GORENG KEMASAN BERMERK 1 (Kg)")  
st.markdown("---")
    
st.markdown("Selamat datang di halaman prediksi harga minyak goreng kemasan bermerk 1 (Kg) di Kota Bandung. Di sini menyajikan prediksi harga minyak goreng menggunakan model machine learning yaitu Long Short Term Memory. Dapatkan informasi harga minyak goreng untuk 100 hari ke depan, mulai dari tanggal 02 Maret 2023.")

st.markdown("Cara Penggunaan")
st.markdown("1. Atur jumlah hari yang ingin Anda prediksi menggunakan slider yang telah tersedia.")
st.markdown("2. Tekan tombol 'Prediksi' untuk melihat grafik dan tabel prediksi harga minyak goreng untuk periode yang Anda pilih")
st.markdown("Saya berharap prediksi ini dapat membantu Anda dalam mengambil keputusan berdasarkan perkiraan harga minyak goreng di masa depan. Silakan nikmati pengalaman menggunakan aplikasi ini!")
st.markdown("Terima kasih telah mengunjungi halaman prediksi harga minyak goreng kami. Selamat Menggunakan!")  
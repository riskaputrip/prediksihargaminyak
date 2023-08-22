import streamlit as st
import json
import streamlit.components.v1 as com
from streamlit_lottie import st_lottie
com.iframe("https://lottie.host/?file=3dc6cf21-a767-4a43-a43d-cb8fc455a4f1/629U4X9K4P.json") 

# Tambahkan CSS untuk menampilkan animasi sejajar dengan teks
st.markdown(
    """
    <style>
    .container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 20vh;
    }
    .text {
        font-size: 18px;
        margin-left: 20px;
        text-align: center
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tulisan di tengah halaman
st.markdown(
    """
    <div style='text-align: center;'>
        <h5>Selamat datang di halaman Kontak saya! Jika Anda memiliki pertanyaan, saran jangan ragu untuk menghubungi melalui kontak di bawah ini. Saya akan dengan senang hati merespons secepat mungkin. Terima kasih atas kunjungan Anda, dan saya berharap dapat membantu Anda dengan yang terbaik!</h5>
        <hr style="border: 1px solid pink; margin: 30px 0;">
        <p>📧 email : riskaputrip20@gmail.com</p>
        <p>📞 No.Telp : 087889120750</p>
    </div>
    """,
    unsafe_allow_html=True
)

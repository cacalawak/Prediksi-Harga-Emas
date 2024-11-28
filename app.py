import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Memuat model yang telah dilatih
@st.cache_resource
def load_model():
    regressor = joblib.load(r'gold_price_model.pkl', 'readwrite')
    return regressor

regressor = load_model()

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.write("Selamat datang di aplikasi prediksi harga emas berbasis web.")
    st.write("<div style='text-align: justify;'>Aplikasi ini menggunakan model prediktif untuk memperkirakan harga emas berdasarkan beberapa variabel pasar seperti indeks S&P 500 (SPX), ETF Emas (GLD), ETF Minyak (USO), dan ETF Perak (SLV). Dengan antarmuka yang sederhana dan mudah digunakan, aplikasi ini dirancang untuk membantu Anda memahami hubungan antar variabel pasar dan memprediksi harga emas dengan cepat dan akurat.</div>", unsafe_allow_html=True)
    st.write("Dibuat oleh Risa Yusnita")

# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset Contoh")
    st.write("Halaman ini biasanya memuat dataset yang digunakan untuk melatih model.")
    # Dataset contoh dapat ditambahkan di sini jika diperlukan
    df = pd.read_csv("gld_price_data.csv")
    st.dataframe(df)
   

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Harga Emas")
    
    # Input pengguna untuk setiap fitur
    spx = st.number_input("Indeks S&P 500 (SPX)", min_value=0.0, max_value=5000.0, value=3000.0)
    gld = st.number_input("ETF Emas (GLD)", min_value=0.0, max_value=500.0, value=120.0)
    uso = st.number_input("ETF Minyak (USO)", min_value=0.0, max_value=200.0, value=40.0)
    slv = st.number_input("ETF Perak (SLV)", min_value=0.0, max_value=100.0, value=15.0)

    # Data input untuk prediksi
    input_data = np.array([[spx, gld, uso, slv]])
    
    # Ketika pengguna mengklik tombol "Prediksi"
    if st.button("Prediksi"):
        prediction = regressor.predict(input_data)
        st.write(f"Perkiraan harga emas adalah: ${prediction[0]:.2f}")

# Fungsi untuk halaman Grafik
def show_grafik():
    st.header("Grafik Harga Emas: Aktual vs Prediksi")
    
    # Data contoh untuk Y_test dan test_data_prediction
    Y_test = [1500, 1550, 1600, 1650, 1700]
    test_data_prediction = [1520, 1570, 1580, 1640, 1680]
    
    # Membuat grafik menggunakan matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test, color='blue', label='Nilai Aktual')
    plt.plot(test_data_prediction, color='green', label='Nilai Prediksi')
    plt.title('Harga Aktual vs Prediksi')
    plt.xlabel('Jumlah Data')
    plt.ylabel('Harga GLD')
    plt.legend()
    plt.grid(True)
    
    # Menampilkan grafik di Streamlit
    st.pyplot(plt)

# Sidebar untuk navigasi
add_selectbox = st.sidebar.selectbox(
    "Pilih Menu",
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

# Logika untuk menampilkan halaman sesuai pilihan
if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()

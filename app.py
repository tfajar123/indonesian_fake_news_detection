import streamlit as st
import pickle
import re
import sqlite3
import requests
from datetime import datetime
import pytz
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Inisialisasi Stemmer dan TF-IDF Vectorizer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
vectorization = TfidfVectorizer()

# Memuat TF-IDF Vectorizer dan Model Machine Learning yang sudah dilatih
vector_form = pickle.load(open('vectorizer.pkl', 'rb'))
naive_bayes_model = pickle.load(open('nb_model.pkl', 'rb'))

# Mengambil stopwords bahasa Indonesia dari NLTK
stop_words = set(stopwords.words('indonesian'))

# Koneksi ke SQLite dan pembuatan tabel
conn = sqlite3.connect('fake_news_log.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT,
                prediction INTEGER,
                probability REAL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
conn.commit()

# Fungsi untuk membersihkan teks dari sumber, link, atau promo
def clean_text(text):
    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Hapus frasa umum yang mengindikasikan sumber
    text = re.sub(r'(Baca artikel .*? selengkapnya)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Download Apps .*? Sekarang)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdi sini:?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Artikel ini telah tayang di Kompas\.com .*? Klik untuk baca: .*?)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Kompascom\+? baca berita tanpa iklan: .*?)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Download aplikasi: .*?)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Fungsi untuk melakukan stemming pada konten input
def stemming(content):
    content = clean_text(content)
    content = re.sub('[^a-zA-Z\s]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [stemmer.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Fungsi untuk sanitasi dan validasi teks input
def validate_text(text):
    if not text.strip():
        return False, "Teks tidak boleh kosong."
    
    # Cek apakah teks terlalu pendek atau hanya berisi karakter acak
    if len(text.split()) < 5 or re.match(r'^[a-zA-Z]{10,}$', text):
        return False, "Teks terlalu pendek atau terlihat tidak bermakna."
    
    # Cek apakah ada script berbahaya
    if re.search(r'<script>|<.*?>|\b(alert|eval|document|window)\b', text, re.IGNORECASE):
        return False, "Teks mengandung elemen berbahaya dan tidak diizinkan."

    return True, "Teks valid."

# Fungsi untuk mengecek apakah teks sudah ada di database
def is_text_in_db(original_text):
    c.execute("SELECT COUNT(*) FROM log WHERE original_text = ?", (original_text,))
    count = c.fetchone()[0]
    return count > 0

# Fungsi untuk menyimpan log ke database hanya jika teks belum ada
def save_to_db(original_text, prediction, probability, model_used):
    cleaned_text = clean_text(original_text)
    if not is_text_in_db(cleaned_text):
        c.execute('''INSERT INTO log (original_text, prediction, probability, model_used)
                     VALUES (?, ?, ?, ?)''', (cleaned_text, prediction, probability, model_used))
        conn.commit()
    # else:
        # st.info("Teks ini sudah pernah diproses sebelumnya, tidak akan disimpan ulang.")


# Fungsi untuk memprediksi berita palsu
def fake_news(news):
    if not news.strip():
        return None, None
    
    news_stemmed = stemming(news)
    input_data = [news_stemmed]
    vector_form1 = vector_form.transform(input_data)

    prediction = naive_bayes_model.predict(vector_form1)
    probability = naive_bayes_model.predict_proba(vector_form1)

    return prediction, probability

# Fungsi untuk mengambil data berita dari API
def get_news(query):
    query = clean_text(query)
    api_url = f"https://serpapi.com/search.json?engine=google_news&q={query}&gl=id&hl=id&api_key=c2bfb9067853a0cc6eb067e5b82276f9eaf2969b93841db3d25d9088d04fa517"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json().get('news_results', [])
    return []

def format_date_to_jakarta(date_str):
    try:
        # Parsing tanggal sesuai format awal
        utc_time = datetime.strptime(date_str, "%m/%d/%Y, %I:%M %p, +0000 UTC")
        
        # Konversi ke zona waktu Jakarta
        utc_zone = pytz.utc
        jakarta_zone = pytz.timezone('Asia/Jakarta')
        
        utc_time = utc_zone.localize(utc_time)
        jakarta_time = utc_time.astimezone(jakarta_zone)

        # Format ulang tanggal
        return jakarta_time.strftime("%d %B %Y, %H:%M WIB")
    except ValueError:
        # Kalau parsing gagal, kembalikan tanggal asli
        return date_str
    
# Fungsi untuk mengunduh log sebagai file CSV atau TXT
def download_logs(format):
    df = pd.read_sql_query("SELECT * FROM log", conn)
    
    if df.empty:
        st.warning("Tidak ada data log untuk diunduh.")
        return
    
    if format == 'CSV':
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Unduh CSV", data=csv, file_name="fake_news_log.csv", mime="text/csv")

# Pengaturan aplikasi Streamlit
def prediction_page():
    st.title('Aplikasi Klasifikasi Berita Palsu')
    st.subheader("Masukkan konten Berita")

    # Text area untuk input pengguna
    sentence = st.text_area("Masukkan konten berita Anda di sini", "", height=200)
    
    # Tombol prediksi
    predict_btt = st.button("Prediksi")

    if predict_btt:
        # Validasi teks input
        valid, message = validate_text(sentence)
        if not valid:
            st.error(message)
        else:
            if not sentence.strip():
                st.error("Silakan masukkan konten berita terlebih dahulu!")
            else:
                prediction_class, probability = fake_news(sentence)
                if prediction_class is not None and probability is not None:
                    prob_fake = probability[0][1] * 100
                    
                    if prediction_class == [0]:
                        st.success(f'Berdasarkan hasil deteksi, Berita ini dapat dipercaya')
                        st.info(f'Kemungkinan berita ini Hoaks: {prob_fake:.2f}%')
                        save_to_db(sentence, 0, prob_fake, 'Naive Bayes')
                    elif prediction_class == [1]:
                        st.error(f'Berdasarkan hasil deteksi, Berita ini Tidak dapat dipercaya')
                        st.info(f'Kemungkinan berita ini Hoaks: {prob_fake:.2f}%')
                        save_to_db(sentence, 1, prob_fake, 'Naive Bayes')
                    else:
                        st.error("Terjadi kesalahan saat melakukan prediksi.")
                    
                    # Ambil berita terkait setelah input valid
                    st.subheader("🔎 Berita Terkait")
                    
                    news_results = get_news(clean_text(sentence))
                    # st.info(clean_text(sentence))
                    
                    if news_results:
                        valid_news = [news for news in news_results if news.get('title') and news.get('link') and news.get('source') and news.get('date')]
                        top_5_news = valid_news[:5]

                        if top_5_news:
                            for news in top_5_news:
                                with st.container():
                                    col1, col2 = st.columns([1, 3])
                                    thumbnail_url = news.get('thumbnail', 'https://via.placeholder.com/150')
                                    title = news['title']
                                    source_name = news['source']['name']
                                    date = news['date']
                                    link = news['link']

                                    with col1:
                                        st.image(thumbnail_url, width=150)
                                    
                                    with col2:
                                        st.markdown(f"### [{title}]({link})")
                                        formatted_date = format_date_to_jakarta(news['date'])
                                        st.write(f"📰 {source_name} | 📅 {formatted_date}")
                                        # st.write(f"[Baca selengkapnya](<{link}>) 🔗")
                                        st.write("---")
                        else:
                            st.warning("Tidak ada berita yang memiliki informasi lengkap.")
                    else:
                        st.warning("Tidak ada berita terkait yang ditemukan.")

                else:
                    st.error("Model tidak valid atau belum dipilih.")

def admin_page():
    st.title('Riwayat Penggunaan Aplikasi')
    st.subheader("Log Prediksi Berita Palsu")

    c.execute("SELECT original_text, prediction, probability, timestamp FROM log ORDER BY timestamp DESC LIMIT 3")
    rows = c.fetchall()

    if rows:
        for row in rows:
            st.write(f"📰 **Teks:** {row[0]}")
            st.write(f"🔍 **Prediksi:** {'Hoaks' if row[1] == 1 else 'Asli'}")
            st.write(f"📊 **Probabilitas:** {row[2]:.2f}%")
            st.write(f"⏱️ **Waktu:** {row[3]}")
            st.write("---")
        
        st.subheader("📂 Unduh Log")
        download_logs('CSV')
        st.warning("DISCLAIMER: Log penggunaan aplikasi masih eksperimental dan hanya digunakan sebagai bahan referensi, tidak digunakan untuk tujuan komersial. Silakan gunakan dengan bijak.")
    else:
        st.info("Belum ada log prediksi yang tersimpan.")

def main():
    st.set_page_config(page_title="Aplikasi Prediksi Berita", initial_sidebar_state="collapsed")
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Prediksi Berita", "Riwayat Penggunaan Aplikasi"])

    if page == "Prediksi Berita":
        prediction_page()
    elif page == "Riwayat Penggunaan Aplikasi":
        admin_page()

if __name__ == '__main__':
    main()

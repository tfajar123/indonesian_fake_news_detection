# 1. Import Library
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

# 2. Load dataset
file_path = '/content/shuffled_data_narasi.csv'  # Sesuaikan path jika berbeda
df = pd.read_csv(file_path, encoding='utf-8')

# 3. Eksplorasi awal data
print(df.head())
print(df.info())
print(df['hoax'].value_counts())

df['text'] = df['Narasi']

# 4. Preprocessing teks untuk Bahasa Indonesia
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    stopwords_indonesia = stopwords.words('indonesian')
except:
    import nltk
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')

# 5.  Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    if pd.isnull(text):
        return ""  # Tangani nilai NaN jadi string kosong
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stopwords_indonesia])
    return text

df['text'] = df['text'].apply(clean_text)

# Cek hasilnya
print(df[['Narasi', 'text']].head())

# 6. Pemisahan data training & testing
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['hoax'], test_size=0.2, random_state=42
)

# 7. Konversi teks ke vektor TF-IDF dengan penyesuaian untuk 9000 data
vectorizer = TfidfVectorizer(max_df=0.9, min_df=10)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

import matplotlib.pyplot as plt

# 8. Training model Naive Bayes
print("\n=== Tuning Hyperparameter Naive Bayes ===")
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
}

grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3, n_jobs=-1, verbose=2)
grid_search_nb.fit(X_train_vec, y_train)

# Model terbaik Naive Bayes
best_nb = grid_search_nb.best_estimator_
print("Best parameters for Naive Bayes:", grid_search_nb.best_params_)

# 9. Test Akurasi, precision, recall, dan f1-score Naive Bayes
y_proba_nb = best_nb.predict_proba(X_test_vec)[:, 1]
threshold = 0.5  # Contoh: Turunkan threshold untuk lebih sensitif terhadap kelas positif
y_pred_nb_threshold = (y_proba_nb >= threshold).astype(int)

print("Akurasi Naive Bayes setelah tuning dan threshold:", accuracy_score(y_test, y_pred_nb_threshold))
print("\nLaporan Klasifikasi Naive Bayes:\n", classification_report(y_test, y_pred_nb_threshold))

# Menampilkan Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb_threshold)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=best_nb.classes_)

disp_nb.plot(cmap='Blues')
plt.title('Confusion Matrix Naive Bayes (Setelah Tuning & Threshold)')
plt.show()

# 10. Simpan model sebagai file .pkl
with open('naive_bayes_model.pkl', 'wb') as nb_file:
    pickle.dump(best_nb, nb_file)

with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model Random Forest dan Naive Bayes telah disimpan sebagai file .pkl!")
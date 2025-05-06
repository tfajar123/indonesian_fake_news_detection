import pickle
import numpy as np

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()

# Contoh: ambil 10 kata acak beserta index-nya
import random
random_features = random.sample(list(feature_names), 10)
for word in random_features:
    index = np.where(feature_names == word)[0][0]
    print(f"{word} → index {index}")

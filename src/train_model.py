import pandas as pd
import pickle
import ast
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Absolute path fix
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

movies_path = os.path.join(BASE_DIR, "dataset", "movies.csv")
credits_path = os.path.join(BASE_DIR, "dataset", "credits.csv")

# Load datasets
movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

# Merge
movies = movies.merge(credits, on='title')

# Select columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

# Convert functions
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i['name'])
    return L

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# Apply conversions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Overview split
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces
for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ","") for i in x])

# Tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Save model
model_path = os.path.join(BASE_DIR, "model")
os.makedirs(model_path, exist_ok=True)

pickle.dump(new_df, open(os.path.join(model_path, "movies.pkl"), 'wb'))
pickle.dump(similarity, open(os.path.join(model_path, "similarity.pkl"), 'wb'))

print("✅ Model trained successfully")
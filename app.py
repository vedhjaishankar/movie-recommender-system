# Vedh Jaishankar
# Uses LensKit 2025.2.0

# app.py
import streamlit as st
import pandas as pd
import requests

# -------------------------------
# TMDb API setup
# -------------------------------
TMDB_API_KEY = 'YOUR_API_KEY'  # Private if published, left my API for grading purposes
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w200'

def fetch_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return TMDB_IMAGE_BASE + poster_path
        return None
    except Exception as e:
        return None

# -------------------------------
# Load MovieLens data
# -------------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")  # userId, movieId, rating, timestamp
    movies = pd.read_csv("ml-latest-small/movies.csv")    # movieId, title, genres
    links = pd.read_csv("ml-latest-small/links.csv")      # movieId, imdbId, tmdbId
    recs_df = pd.read_csv("recs.csv")                     # user_id, item_id, score
    return ratings, movies, links, recs_df

ratings, movies, links, recs_df = load_data()

# Merge movies with links
movies = movies.merge(links[['movieId','tmdbId']], on='movieId', how='left')

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="MovieLens Recommender", layout="wide")
st.title("🎬 MovieLens ALS Recommender System")

st.sidebar.header("User Controls")
unique_users = ratings['userId'].unique()
selected_user = st.sidebar.selectbox("Select a user:", unique_users)

min_score = st.sidebar.slider("Minimum predicted score for recommendations:", 0.0, 5.0, 0.0, 0.1)
num_movies = st.sidebar.slider("Number of recommendations to display:", 1, 100, 10)

# -------------------------------
# Filter recommendations for selected user
# -------------------------------
user_recs = recs_df[recs_df['user_id'] == selected_user]
user_recs = user_recs[user_recs['score'] >= min_score]
user_recs = user_recs.merge(movies, left_on="item_id", right_on="movieId")
user_recs = user_recs.sort_values("score", ascending=False).head(num_movies)

# -------------------------------
# Display recommendations in card style
# -------------------------------
st.subheader(f"Top {num_movies} Recommendations for User {selected_user}")

cols = st.columns(5)  # Display 5 cards per row
for i, (_, row) in enumerate(user_recs.iterrows()):
    col = cols[i % 5]
    poster_url = fetch_poster_url(row['tmdbId'])
    if poster_url:
        col.image(poster_url)
    col.markdown(f"**{row['title']}**  \nScore: {row['score']:.2f}")

# -------------------------------
# Display user's rated movies
# -------------------------------
st.subheader(f"Movies Already Rated by User {selected_user}")
user_history = ratings[ratings['userId'] == selected_user]
user_history = user_history.merge(movies, on="movieId")[['title','rating']]
user_history = user_history.sort_values('rating', ascending=False)
st.dataframe(user_history, use_container_width=True)

# -------------------------------
# Search movies in recommendations
# -------------------------------
st.subheader("🔍 Search Movies in Recommendations")
search_query = st.text_input("Search movie title:")
if search_query:
    filtered = user_recs[user_recs['title'].str.contains(search_query, case=False)]
    st.dataframe(filtered[['title','score']], use_container_width=True)

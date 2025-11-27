# Movie Recommender System (LensKit + Streamlit)

A full Movie Recommender System built using LensKit 2025.2.0, Matrix Factorization (ALS), and a Streamlit web app for interactive recommendations. The system trains an ALS model on the MovieLens dataset, evaluates it with ranking metrics (NDCG, Precision, Recall), and displays personalized recommendations with movie posters pulled from the TMDb API.

## Features
### Backend / Model Training
- Uses LensKit 2025.2.0 for recommender modeling
- Matrix Factorization (BiasedMF) with 65 latent features
- 5-fold cross-user validation via LensKit's `crossfold_users`
- Computes top-N recommendations for each user
- Evaluates model using:
  - NDCG@10
  - Precision@10
  - Recall@10
- Exported CSV outputs for reproducibility (`ratings.csv`, `movies.csv`, `recs.csv`, `metrics.csv`)

### 🎨 Frontend (Streamlit Web App)
- Full interactive UI built using **Streamlit**
- Select any MovieLens user and view:
  - Top-N recommended movies
  - Previously rated movies
- Card‑style movie display with posters from **TMDb API**
- Filters: minimum score threshold, number of movies to display
- Search bar for filtering recommendations

---

## Model Training Overview
### 1. Load MovieLens Data
- Ratings: userId, movieId, rating, timestamp
- Movies: movieId, title, genres
- Optional: Links for TMDb IDs

### 2. Build the Dataset in LensKit
Converted using `from_interactions_df`.

### 3. Crossfold Evaluation
```
for split in crossfold_users(dataset, 5, SampleFrac(1)):
    train ALS
    generate top‑100 recs
    collect metrics
```

### 4. Evaluation Results (Your Output)
- **Avg NDCG@10:** ~0.25
- **Avg Precision@10:** ~0.22
- **Avg Recall@10:** ~0.22

---

## 🎬 Streamlit Web App Overview
### User Features
- Select a user from sidebar  
- Display top recommendations  
- Show movie posters (TMDb API)  
- Search recommendations  
- Explore user rating history

### TMDb Integration
Movie posters fetched via:
```
https://api.themoviedb.org/3/movie/{tmdb_id}
```
---

## Future Improvements
- Add hybrid (content + collaborative) modeling
- Replace BiasedMF with Neural Collaborative Filtering
- Add user login + saving preferences
- Deploy via Streamlit Cloud / Docker

---
Built using LensKit 2025.2.0 and MovieLens dataset.


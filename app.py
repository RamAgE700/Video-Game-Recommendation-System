import streamlit as st
import pandas as pd
import pickle

# Load the trained k-NN model and the processed dataset
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    features = pickle.load(f)

df = pd.read_csv('video_game_data_processed.csv')

st.title('Video Game Recommendation System')

# Allow user to select a genre
genres = df['Genres'].str.get_dummies(sep=',').columns.tolist()
selected_genre = st.selectbox('Select Genre', genres)

if st.button('Get Recommendations'):
    # Filter games of the selected genre
    genre_games = df[df[selected_genre] == 1]
    
    if len(genre_games) > 0:
        # Find similar games using k-NN
        distances, indices = knn.kneighbors(genre_games[features], n_neighbors=10)
        
        recommendations = df.iloc[indices[0]][['Title', 'Price', 'Rating']]
        st.write(f"Top {len(recommendations)} recommendations for {selected_genre} genre:")
        st.table(recommendations)
    else:
        st.write(f"No games found for genre {selected_genre}.")
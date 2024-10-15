import streamlit as st
import joblib
import pandas as pd

# Function to load the content-based model components
def load_content_based_model():
    # Load the CountVectorizer object
    count_vectorizer = joblib.load('model/count_vectorizer.pkl')

    # Load the cosine similarity matrix
    cosine_sim = joblib.load('model/cosine_sim_matrix.pkl')
    
    return count_vectorizer, cosine_sim

# Function to load the collaborative filtering model (SVD)
def load_collaborative_model():
    # Load Surprise SVD model
    svd_model = joblib.load('model/svd_recommender_model.pkl')  # No unpacking
    return svd_model

# Hybrid recommendation logic (content-based and collaborative filtering)
def hybrid_recommendations(userId, title, cosine_sim, sdf, indices, indices_map, svd_model):
    # Ensure the title is correctly capitalized
    title = str(title).title()
    
    # Get the index of the title from the indices
    idx = indices[title]
    
    # Get pairwise similarity scores of all movies with the chosen movie
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 25 most similar movies
    sim_scores = sim_scores[1:26]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Create a DataFrame for the 25 most similar movies
    movies = sdf.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    
    # Apply collaborative filtering to re-rank the movies
    movies['est'] = movies['id'].apply(lambda x: svd_model.predict(userId, indices_map.loc[x]['movieId']).est)
    
    # Sort the movies based on the estimated rating from the SVD model
    movies = movies.sort_values('est', ascending=False)
    
    # Return the top 10 recommended movies
    return movies.head(10)

# Streamlit App
st.title("Hybrid Recommender System")

# Load models
count_vectorizer, cosine_sim = load_content_based_model()
svd_model = load_collaborative_model()

# Load the dataset (sdf) and indices for movie titles
sdf = pd.read_csv('dataset/sdf_dataset.csv')

indices_map = joblib.load('model/indices_map.pkl')
indices = joblib.load('model/indices_series.pkl')

# User input: Enter preferences or movie title and userId
user_id_input = st.number_input("Enter your user ID:", min_value=1, step=1)
movie_input = st.text_input("Enter the movie title:")

# Button to generate recommendations
if st.button("Recommend"):
    if movie_input:
        # Generate hybrid recommendations using the provided user ID and movie title
        recommendations = hybrid_recommendations(user_id_input, movie_input, cosine_sim, sdf, indices, indices_map, svd_model)
        
        # Display recommendations
        st.write("Top recommendations for you:")
        st.write(recommendations[['title', 'year', 'vote_average', 'est']])
    else:
        st.write("Please enter a movie title.")

# streamlit run hybrid_recommender.py
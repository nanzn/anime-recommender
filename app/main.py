# SETUP
# Libraries 
import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp
import operator 

from sklearn.metrics.pairwise import cosine_similarity

# Constants
DATA_PATH = "./data/"

# LOAD DATA
df = pd.read_csv(f"{DATA_PATH}anime.csv")
print(df.head())


# Step 3: Data Collection

# Download a movie dataset (e.g., from Kaggle) in CSV format and save it in the project directory.
# Load the dataset into a pandas DataFrame:
# python
# Copy code
# df_movies = pd.read_csv("movies.csv")
# Step 4: Data Preprocessing

# Clean the dataset by handling missing values, duplicate entries, and irrelevant columns.
# Transform the dataset into a suitable format for recommendation purposes. For example, create a matrix where each row represents a movie and each column represents a user's rating for that movie.
# Step 5: Building the Recommendation Engine

# Choose a recommendation algorithm, such as collaborative filtering or content-based filtering.
# Implement the recommendation logic using the selected algorithm. For example, you can use the cosine similarity measure to find similar movies based on user ratings.
# python
# Copy code
# def get_movie_recommendations(movie_title, num_recommendations=5):
#     # Retrieve the index of the movie
#     movie_index = df_movies[df_movies["title"] == movie_title].index[0]
    
#     # Calculate cosine similarity between the movie and all other movies
#     similarity_scores = cosine_similarity(matrix, matrix[movie_index])
    
#     # Get the indices of movies with highest similarity scores
#     similar_movies_indices = similarity_scores.argsort(axis=0)[-num_recommendations-1:-1][::-1]
    
#     # Return the recommended movie titles
#     return df_movies["title"].iloc[similar_movies_indices]
# Step 6: Creating the Web Application

# Use Streamlit to create the user interface for the web application:
# python
# Copy code
# def main():
#     st.title("Movie Recommendation System")
    
#     movie_title = st.text_input("Enter a movie title:")
#     if st.button("Recommend"):
#         recommendations = get_movie_recommendations(movie_title)
#         st.subheader("Recommended Movies:")
#         for movie in recommendations:
#             st.write(movie)
# Step 7: Integrating the Recommendation Engine

# Integrate the recommendation engine code with the Streamlit application:
# python
# Copy code
# if __name__ == "__main__":
#     main()
# Step 8: Running the Web Application

# In the terminal, navigate to the project directory and run the Streamlit application using the command streamlit run movie_recommender.py.
# The web application will open in a browser, allowing you to test the recommendation system by entering movie titles and clicking the "Recommend" button.
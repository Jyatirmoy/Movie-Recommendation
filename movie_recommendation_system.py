import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge the datasets
data = pd.merge(ratings, movies, on='movieId')

# Prepare the data for Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD for collaborative filtering
algo = SVD()
algo.fit(trainset)

# Evaluate the algorithm
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Function to get top N recommendations for a user
def get_recommendations(user_id, n=10):
    # Get a list of all movie ids
    movie_ids = ratings['movieId'].unique()
    
    # Get predictions for the user
    user_ratings = [(movie_id, algo.predict(user_id, movie_id).est) for movie_id in movie_ids]
    
    # Sort by rating and get the top N
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n = user_ratings[:n]
    
    # Get movie titles for the top N movie ids
    top_n_movies = [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id, _ in top_n]
    
    return top_n_movies

# Get recommendations for a user
user_id = 1
recommendations = get_recommendations(user_id)
print("Top 10 movie recommendations for user", user_id, "are:")
for i, movie in enumerate(recommendations):
    print(f"{i+1}. {movie}")

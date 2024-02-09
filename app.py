#may require to run this before running app.py in terminal
#pip install --upgrade watchdog
#pip install --upgrade Flask flask_pymongo flask_cors
# import Flask

from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS


# knn imports
# Importing Libraries
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.sparse import csr_matrix

# Create an app, being sure to pass __name__
app = Flask(__name__)
CORS(app)
# configure flask to connect to MongoDB

app.config['MONGO_URI'] = 'mongodb://localhost:27017/movies_database'
mongo = PyMongo(app)

#Define what to do when a user hits the index route
@app.route("/")
def home():
     """List all available api routes."""
     return (
        f"Available Routes:<br/>"
        f"/api/v1.0/movies_list<br/>"
        
    )
     
def create_matrix(df):
     
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
     
    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
     
    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
     
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
 
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
     
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

#Define what to do when a user hits the /about route
@app.route("/api/v1.0/movies_list/")
def get_population_data():
    movie_collection = mongo.db.movies_list
    data = list(movie_collection.find({}, {'_id': 0}))
    return (data)

# Query parameter is user_id
@app.route("/api/v1.0/movie_recommendations")
def get_movie_recommendations():
    user_id = int(request.args.get("user_id"))  # Grab user_id from API query parameter

    # Part 1: KNN Model
    movie_recommendation_ids = get_recommendations_with_knn_model(user_id)
    print('movie recommendation ids: ', movie_recommendation_ids)
    # Part 2: Fetch movies recommended by KNN model from mongodb
    movie_collection = mongo.db.movies_list
    # movie_recommendation_data = list(movie_collection.find({}, {'_id': 0}))
    # print(movie_recommendation_data)
    
    # return final list of movie recommendations
    movie_recommendations = {};
    return (movie_recommendations)

def get_recommendations_with_knn_model(user_id):
    movie_collection = mongo.db.movies_list
    data = list(movie_collection.find({}, {'_id': 0}))
    
    movies_combined_df = pd.DataFrame(data)
    movies_df = movies_combined_df[['movieId', 'title', 'cleaned_genres']].copy()
    ratings_df = movies_combined_df[['userId', 'movieId', 'rating', 'timestamp']].copy()
    
    n_ratings = len(ratings_df)
    n_movies = len(ratings_df['movieId'].unique())
    n_users = len(ratings_df['userId'].unique())
    
    print(f"Number of ratings: {n_ratings}")
    print(f"Number of unique movieId's: {n_movies}")
    print(f"Number of unique users: {n_users}")
    print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
    print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

    user_freq = ratings_df[['userId', 'movieId']].groupby('userId').count().reset_index()
    user_freq.columns = ['userId', 'n_ratings']
    print(user_freq.head())

    # Find Lowest and Highest rated movies:
    mean_rating = ratings_df.groupby('movieId')[['rating']].mean()
    # Lowest rated movies
    lowest_rated = mean_rating['rating'].idxmin()
    movies_df.loc[movies_df['movieId'] == lowest_rated]
    # Highest rated movies
    highest_rated = mean_rating['rating'].idxmax()
    movies_df.loc[movies_df['movieId'] == highest_rated]
    # show number of people who rated movies rated movie highest
    ratings_df[ratings_df['movieId']==highest_rated]
    # show number of people who rated movies rated movie lowest
    ratings_df[ratings_df['movieId']==lowest_rated]
    
    ## the above movies has very low dataset. We will use bayesian average
    movie_stats = ratings_df.groupby('movieId')[['rating']].agg(['count', 'mean'])
    movie_stats.columns = movie_stats.columns.droplevel()

    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings_df)

    def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
        
        neighbour_ids = []
        
        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        k+=1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(X)
        movie_vec = movie_vec.reshape(1,-1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
        for i in range(0,k):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
        df1 = ratings_df[ratings_df['userId'] == user_id]
        
        if df1.empty:
            print(f"User with ID {user_id} does not exist.")
            return
    
        movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    
        movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
    
        similar_ids = find_similar_movies(movie_id, X, k)
        movie_title = movie_titles.get(movie_id, "Movie not found")
    
        if movie_title == "Movie not found":
            print(f"Movie with ID {movie_id} not found.")
            return
    
        print(f"Since you watched {movie_title}, you might also like:")
        for i in similar_ids:
            print(movie_titles.get(i, "Movie not found"))

        return similar_ids
    
    movie_recommendation_ids = recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
    return movie_recommendation_ids


if __name__ == "__main__":
    app.run(debug=True)

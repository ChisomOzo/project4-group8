#may require to run this before running app.py in terminal
#pip install --upgrade watchdog
#pip install --upgrade Flask flask_pymongo flask_cors
# import Flask

from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS


# imports
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
import random
import math
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create an app, being sure to pass __name__
app = Flask(__name__)
# Updated CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
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

#Define what to do when a user hits the /about route
@app.route("/api/v1.0/movies_list/")
def get_population_data():
    movie_collection = mongo.db.movies_list
    data = list(movie_collection.find({}, {'_id': 0}))
    return (data)


#class CFModel(tf.keras.Model):
 #   def __init__(self, n_users, m_items, k_factors):
 #       super(CFModel, self).__init__()
  #      
   #     self.P = tf.keras.Sequential([
    #        tf.keras.layers.Embedding(n_users, k_factors, input_length=1),
     #       tf.keras.layers.Reshape((k_factors,))
      #  ])
        
       # self.Q = tf.keras.Sequential([
        #    tf.keras.layers.Embedding(m_items, k_factors, input_length=1),
         #   tf.keras.layers.Reshape((k_factors,))
        #])
        
    #def call(self, inputs):
    #    user_id, item_id = inputs
    #    user_latent = self.P(user_id)
    #    item_latent = self.Q(item_id)
    #    return tf.reduce_sum(tf.multiply(user_latent, item_latent), axis=1)
    
    #def rate(self, user_id, item_id):
    #    user_embedding = self.P(tf.constant([user_id]))
    #    item_embedding = self.Q(tf.constant([item_id]))
    #    prediction = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=1)[0]
    #    return prediction.numpy()  

# Deep Learning Model -- START --
#@app.route("/api/v2.0/movie_recommendations")
#def get_movie_recommendations_v2():
#    user_id = int(request.args.get("user_id"))  # Grab user_id from API query parameter
    
    # Part 1: Deep Learning Model
#    movie_recommendation_ids = get_recommendations_with_deep_learning_model(user_id)
#    movie_recommendation_ids = list(map(lambda n: int(n), movie_recommendation_ids))
#    print('movie recommendation ids: ', movie_recommendation_ids)

    # Part 2: Fetch movies recommended by deep learning model from mongodb
#    movies_details_collection = mongo.db.movies_details_list
#    movie_recommendation_data = list(movies_details_collection.find({'movieId': {'$in': movie_recommendation_ids}}, {'_id': 0}))
#    print('movie_recommendation_data: ', movie_recommendation_data)
    
    # return list of movies back to front-end
#    response = {'data': movie_recommendation_data}
#    return response

#def get_recommendations_with_deep_learning_model(user_id):
#    movie_collection = mongo.db.movies_list
#    data = list(movie_collection.find({}, {'_id': 0}))

#    movies_combined_df = pd.DataFrame(data)
#    n_ratings = len(movies_combined_df)
#    n_movies = len(movies_combined_df['movieId'].unique())
#    n_users = len(movies_combined_df['userId'].unique())
    
#    print(f"Number of ratings: {n_ratings}")
#    print(f"Number of unique movieId's: {n_movies}")
#    print(f"Number of unique users: {n_users}")
#    print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
#    print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

    # Select columns and drop duplicates
#    movies_df = movies_combined_df[['movieId', 'title', 'cleaned_genres']].copy()
#    movies_df = movies_df.drop_duplicates()

    # Check the number of ratings and unique movieId's
#    n_ratings = len(movies_df)
#    n_movies = len(movies_df['movieId'].unique())

#    print(f"Number of ratings: {n_ratings}")
#    print(f"Number of unique movieId's: {n_movies}")

    # Select columns
#    ratings_df = movies_combined_df[['userId', 'movieId', 'rating']].copy()

    # Check the number of ratings, unique movieId's, and unique users
#    n_ratings = len(ratings_df)
#    n_movies = len(ratings_df['movieId'].unique())
#    n_users = len(ratings_df['userId'].unique())
    
#    print(f"Number of ratings: {n_ratings}")
#    print(f"Number of unique movieId's: {n_movies}")
#    print(f"Number of unique users: {n_users}")

    # Select columns and drop duplicates
#    users_df = movies_combined_df[['userId', 'gender', 'zipcode', 'age_desc', 'occ_desc']] .copy()
#    users_df = users_df.drop_duplicates()

    # Check the number of ratings and unique users
#    n_ratings = len(users_df)
#    n_users = len(users_df['userId'].unique())
    
#    print(f"Number of ratings: {n_ratings}")
#    print(f"Number of unique users: {n_users}")

#    RNG_SEED = 142
#    random_ratings = ratings_df.sample(frac=1, random_state=RNG_SEED)

    # Randomize the dataframes
#    users = random_ratings['userId'].values
#    movies = random_ratings['movieId'].values
#    ratings = random_ratings['rating'].values

#    print(f"users:", users, ', shape =', users.shape)
#    print(f"movies:", movies, ', shape =', movies.shape)
#    print(f"ratings:", ratings, ', shape =', ratings.shape)

    # Capture the max userId and movieId
#    user_id_max = ratings_df['userId'].drop_duplicates().max()
#    movie_id_max = ratings_df['movieId'].drop_duplicates().max()

    # Ensure user_id and movie_id fall within range
#    n_users = user_id_max + 1
#    m_items = movie_id_max + 1
    
    # Test with constant 
#    FACTORS = 100

    # Colabritive filtering model
#    cf_model = CFModel(n_users, m_items, FACTORS)

    # Compile, loss: Mean Squared Error, opimizer: Adamax
#    cf_model.compile(loss='mse', optimizer='adamax')

    # Ensure compatibility with model
#    print("Max User ID:", user_id_max)
#    print("Max Movie ID:", movie_id_max)
#    print("Number of users (n_users):", n_users)
#    print("Number of items (m_items):", m_items)

    # Train the model
    # Set callbacks to monitor validation loss and save the best model weights
#    callbacks = [
#        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
#    ]

    # Use epochs for training
#    epochs = 1

    # Fit the model with callbacks
#    results = cf_model.fit(
#        x=[users, movies],
#       y=ratings,
#        epochs=epochs,
#        validation_split=.1,
#        verbose=2,
#        callbacks=callbacks
#    )

    # Save the entire model using the TensorFlow SavedModel format
    # tf.keras.models.save_model(cf_model, 'saved_model')

    # Show the best validation RMSE
#    val_losses = results.history['val_loss']
#    min_val_loss = min(val_losses)
#    idx = val_losses.index(min_val_loss) + 1  # Add 1 to get epoch number starting from 1
#    print('Minimum RMSE at epoch', idx, '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

    # Define predicted rating
#    def predict_rating(userId, movieId):
#        return cf_model.rate(userId - 1, movieId - 1)

    # Get the top rated movies by current user
#    user_ratings = ratings_df[ratings_df["userId"] == user_id][['userId', 'movieId', 'rating']]
#    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(user_id, x['movieId']), axis=1)

    # Remove duplicate movie entries from movies_df DataFrame
#    unique_movies_df = movies_df.drop_duplicates(subset=['movieId'])

    # Merge the user_ratings DataFrame with the unique_movies_df DataFrame to get the top 10 movies by current user.
#    merged_data = user_ratings.merge(unique_movies_df, on='movieId', how='inner')

    # Sort the DataFrame by rating in descending order and reset the index.
#    top_10_rated_movies = merged_data.sort_values(by='rating', ascending=False)
#    top_10_rated_movies = top_10_rated_movies.reset_index(drop=True)


    # Get unrated movies for the user
#    recommendations = ratings_df[ratings_df['movieId'].isin(user_ratings['movieId'])== False][['movieId']].drop_duplicates()

    # Generate predictions for unrated movies
#    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(user_id, x['movieId']), axis=1)

    # Merge predictions with movie information
#    recommended_movies = recommendations.sort_values(by='prediction', ascending=False).merge(movies_df,
#                                                                                            on='movieId',
#                                                                                            how='inner',
#                                                                                            suffixes=['_u', '_m'])

    # Filter out duplicate titles for one specific user
#    recommended_movies = recommended_movies.drop_duplicates(subset=['title'])
    # Reset index
#    top_10_recommended_movies = recommended_movies.reset_index(drop=True)
#    top_10_recommended_movieIds = list(top_10_recommended_movies['movieId'].head(10).values)

#    print('top_10_recommended_movies: ', top_10_recommended_movieIds)

#    return top_10_recommended_movieIds


# Deep Learning Model -- END --

# KNN Model -- START --
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

# Query parameter is user_id
@app.route("/api/v1.0/movie_recommendations")
def get_movie_recommendations():
    user_id = int(request.args.get("user_id"))  # Grab user_id from API query parameter

    # Part 1: KNN Model
    movie_recommendation_ids = get_recommendations_with_knn_model(user_id)
    print('movie recommendation ids: ', movie_recommendation_ids)

    movie_recommendation_ids = list(map(lambda n: int(n), movie_recommendation_ids))

    # Part 2: Fetch movies recommended by KNN model from mongodb
    movies_details_collection = mongo.db.movies_details_list
    movie_recommendation_data = list(movies_details_collection.find({'movieId': {'$in': movie_recommendation_ids}}, {'_id': 0}))
    print('movie_recommendation_data: ', movie_recommendation_data)
    # print(movie_recommendation_data)
    
    # return final list of movie recommendations
    movie_recommendations = {'data': movie_recommendation_data}
    return movie_recommendations

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

# KNN Model -- END --

if __name__ == "__main__":
    app.run(debug=False)

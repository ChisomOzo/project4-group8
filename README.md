# Movie Recommendation  Project

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Sources](#data-sources)
- [Data Cleanup and Analysis](#data-cleanup-and-analysis)
- [Flask API](#flask-api)
- [Frontend Development](#frontend-development)
- [Data Visualization](#data-visualization)
- [Contributors](#contributors)
- [References](#references)


## Project Overview
Welcome to the Movie Recommendation Project! This comprehensive project recommends movies to users based on their favourite movie that was watched by users in the past. Leveraging a Flask API to serve data to the frontend, the project's user interface is developed using the deep learning model. The dataset used had users’ rating on past movies watched, using those ratings we were able to create a model that would predict the movie that the user will watch and provide a high rating.

## Project Structure
The project is structured as follows:
- `app.py`: Contains the Flask API implementation.
- `index.html`: For HTML templates.
- In the file MONGO_DB we have data base connnection and in movies_clean and users_clean we have the EDA


## Data Cleanup and Analysis:
Data cleanup and analysis are performed in visual studio code. Explore, clean, and reformat data to prepare it for analysis. The data required the merging of different sets of data which includes users' id, movie ratings, genre and overview. The joint data was a merge of movie_metadata.csv and rating_small.csv, this allowed us to drop any movie that had a rating but with no user id. 

In the data cleaning process, we had to clean the genre column as it was in a list of dictionary format, for example "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]'" after the cleaning the format changed to "Animation|Comedy|Family"

## Flask API
The Flask API (`app.py`) serves as the backend for the project. It handles data retrieval and user-driven interactions. 


## Frontend Development
The frontend portion of this project was built by using HTML user-driven interactions, which are implemented using a JS library and CSS.
The html includes a text input bar where users can enter their user ID number and click search or hit enter to run the Deep Learning model and generate movie recommendations for that user.
If numbers outside the user range are entered or non-integer characters are entered, an error message will print.
A load spinner from https://cssloaders.github.io/ was added to indicate that the backend model is loading.
Once finished, the model output displays the top 10 movie recommendations for that given user.

## Data Visualization
Data visualizations are seamlessly integrated. These visualizations weave a compelling list of movies and genres to choose from. 
Once a user ID is entered, the model outcome will generate a list of the top 10 movie recommendations for that given user. On the html, the list will display in 2 columns, where the movie posters, title of the movie, genres, and overview will also display. 
For movie poster paths that did not work, a general grey image was displayed instead. 

## Machine Learning Models
K-Nearest Neighbors (KNN) Approach:

Data Retrieval and Preprocessing:
It fetches movie data from an API endpoint and preprocesses it by extracting relevant columns (movieId, title, cleaned_genres) to create a movies DataFrame.
It extracts relevant columns (userId, movieId, rating) and drops duplicate entries to create a ratings DataFrame.

Data Analysis:
It calculates basic statistics such as the number of ratings, unique movie IDs, unique users, etc.
It analyzes user frequency and identifies the lowest and highest rated movies.

User-Item Matrix Creation:
It creates a sparse user-item matrix using SciPy's csr_matrix function.

Finding Similar Movies:
It defines a function find_similar_movies that finds similar movies based on a given movie ID using the KNN algorithm with cosine similarity metric.

Recommendation for a Specific User:
It defines a function recommend_movies_for_user that recommends movies for a specific user based on the movies they have watched and their ratings. This function utilizes the KNN-based similarity to suggest similar movies.

Custom TensorFlow Model Approach:

Data Retrieval and Preprocessing:
Similar to the KNN approach, it fetches and preprocesses movie data to create movies, ratings, and users DataFrames.

Model Initialization:
It defines a collaborative filtering model using TensorFlow, comprising embeddings for users and movies.

Model Training:
It trains the collaborative filtering model using the ratings data, using techniques like early stopping to prevent overfitting and save the best model.

Model Evaluation:
It evaluates the model's performance by calculating the minimum Root Mean Squared Error (RMSE) on the validation set.

Recommendation Generation:
It generates recommendations for a specific user by predicting ratings for unrated movies using the trained model.
It retrieves the top-rated movies and the top recommended movies for the specified user based on predicted ratings.

## Contributors
- Angad Dhillon
- Camille Velarde
- Chandler McLaren
- Chisom Ozoemena

## Python Libraies
- [Tensor Flow](https://www.tensorflow.org/)
- [Skit-Learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Java Script D3](https://d3js.org/)
- [Mongo_DB](https://mongodb.github.io/node-mongodb-native/api-generated/mongoclient.html)

## References
- [Movie Dataset]([https://flask.palletsprojects.com/](https://github.com/khanhnamle1994/movielens/tree/master))
- [Java Script]([https://leafletjs.com/](https://getbootstrap.com/docs/3.3/javascript/))
- [Data Source](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- [Reference 1](https://www.geeksforgeeks.org/recommendation-system-in-python/)
- [Reference 2]((https://github.com/khanhnamle1994/movielens/tree/master)https://github.com/khanhnamle1994/movielens/tree/master)
- [Reference 3]([https://www.geeksforgeeks.org/recommendation-system-in-python/](https://cssloaders.github.io/)https://cssloaders.github.io/)







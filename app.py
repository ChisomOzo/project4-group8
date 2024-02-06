#may require to run this before running app.py in terminal
#pip install --upgrade watchdog
#pip install --upgrade Flask flask_pymongo flask_cors
# import Flask

from flask import Flask, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS

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

     


#Define what to do when a user hits the /about route
@app.route("/api/v1.0/movies_list")
def get_population_data():
    population_collection = mongo.db.movies_list
    data = list(population_collection.find({}, {'_id': 0}))
    return (data)




if __name__ == "__main__":
    app.run(debug=True)

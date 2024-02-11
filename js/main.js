const recommendationUrl = "http://localhost:5000/api/v1.0/movie_recommendations?user_id=";

function sendMovieRecommendationRequest() {
    console.log('hit search button')
    let userIdinput = "";
    //TODO: grab text from html input element and store into movieInput object

    const input = document.getElementById("userId-input")
    userIdinput = input.value;
    console.log(userIdinput);
    
    const searchStatement = document.getElementById("search-statement");
    searchStatement.style.display = 'block';
    searchStatement.textContent = 'Searching movie recommendations for User ID: "' + userIdinput + '"';

    // Send the API request
    d3.json(recommendationUrl + userIdinput).then(loaddata);

    // clear input box after search
    input.value = "";

}

document.getElementById("search-button").onclick = sendMovieRecommendationRequest;

var userinputElement = document.getElementById("userId-input");
userinputElement.addEventListener("keydown", function (e) {
    if (e.code === "Enter") {  //checks whether the pressed key is "Enter"
        sendMovieRecommendationRequest();
    }
});


// loaddata(dummyData);

function loaddata(response) {
    // 1. Clear movie recommendation div
    clearRecommendations();

    // 2. Load the movie recommendations from the response.data 
    var recommendationDiv = document.getElementById('recommendations-container');
    let recommendations = response.data;
    recommendations.forEach(recommendation => {
        let movieContainer = document.createElement('div'); 
        movieContainer.className="movie-container";
        movieContainer.innerHTML = `<img class="movie-image" src="https://image.tmdb.org/t/p/w500/${recommendation.poster_path}" onerror="this.src='img/movie-placeholder.jpg';"/>
        
        <div class="movie-text">
            <h2>${recommendation.title}</h2>
            <p>${recommendation.genres}</p>
            <p>${recommendation.overview}</p>
        </div>`;
        recommendationDiv.appendChild(movieContainer);
    });
}

function clearRecommendations() {
    var recommendationDiv = document.getElementById('recommendations-container');
    while(recommendationDiv.firstChild) {
        recommendationDiv.removeChild(recommendationDiv.lastChild);
    }
}

// clearRecommendations();
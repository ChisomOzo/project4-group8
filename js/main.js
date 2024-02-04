console.log('Testing');

dummyData = {
    data: [
        {
            'name': 'GoldenEye',
            'poster_path': '/5c0ovjT41KnYIHYuF4AWsTe3sKh.jpg',
            'genres': 'Action|Thriller',
            'overview': "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences."
        },
        {
            'name': 'Jumanji',
            'poster_path': '/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg',
            'genres': 'Adventure|Fantasy|Comedy',
            'overview': "James Bond must unmask the mysterious head of the Janus Syndicate and prevent the leader from utilizing the GoldenEye weapons system to inflict devastating revenge on Britain."
        },
        {
            'name': 'Big Bully',
            'poster_path': '/wtFekK9VKWujQW9R9c6sYbir6wm.jpg',
            'genres': 'Comedy|Family',
            'overview': "A writer returns to his hometown where he faces the childhood nemesis whose life he ultimately ruined, only the bully wants to relive their painful past by torturing him once again."
        },
        {
            'name': 'Apollo 13',
            'poster_path': '/6JQ9z3V9x4vlU2GSZx2yNO0PvuX.jpg',
            'genres': 'Drama',
            'overview': "The true story of technical troubles that scuttle the Apollo 13 lunar mission in 1971, risking the lives of astronaut Jim Lovell and his crew, with the failed journey turning into a thrilling saga of heroism. Drifting more than 200,000 miles from Earth, the astronauts work furiously with the ground crew to avert tragedy."
        },
        {
            'name': 'A Goofy Movie',
            'poster_path': '/bycmMhO3iIoEDzP768sUjq2RV4T.jpg',
            'genres': 'Romance|Animation|Family|Comedy|Adventure',
            'overview': "Though Goofy always means well, his amiable cluelessness and klutzy pratfalls regularly embarrass his awkward adolescent son, Max. When Max's lighthearted prank on his high-school principal finally gets his longtime crush, Roxanne, to notice him, he asks her on a date. Max's trouble at school convinces Goofy that he and the boy need to bond over a cross-country fishing trip like the one he took with his dad when he was Max's age, which throws a kink in his son's plans to impress Roxanne."
        },
        {
            'name': 'Jurassic Park',
            'poster_path': '/c414cDeQ9b6qLPLeKmiJuLDUREJ.jpg',
            'genres': 'Adventure|Sci-Fi',
            'overview': "A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA. Before opening day, he invites a team of experts and his two eager grandchildren to experience the park and help calm anxious investors. However, the park is anything but amusing as the security systems go off-line and the dinosaurs escape."
        },
        {
            'name': 'Super Mario Bros.',
            'poster_path': '/bmv7fmcBFzjnJvirfAdZm75qERY.jpg',
            'genres': 'Adventure|Comedy|Family|Fantasy',
            'overview': "Mario and Luigi, plumbers from Brooklyn, find themselves in an alternate universe where evolved dinosaurs live in hi-tech squalor. They're the only hope to save our universe from invasion by the dino dictator, Koopa."
        },
        {
            'name': 'Twilight',
            'poster_path': '/nlvPMLCdum7bkHKmDSMnNLGztmW.jpg',
            'genres': 'Adventure|Fantasy|Drama|Romance',
            'overview': "When Bella Swan moves to a small town in the Pacific Northwest to live with her father, she starts school and meets the reclusive Edward Cullen, a mysterious classmate who reveals himself to be a 108-year-old vampire. Despite Edward's repeated cautions, Bella can't help but fall in love with him, a fatal move that endangers her own life when a coven of bloodsuckers try to challenge the Cullen clan."
        },
        {
            'name': 'Bolt',
            'poster_path': '/pGA75RbbIDXlUJtzJogAtS0KyxL.jpg',
            'genres': 'Animation|Family|Adventure|Comedy',
            'overview': "Bolt is the star of the biggest show in Hollywood. The only problem is, he thinks it's real. After he's accidentally shipped to New York City and separated from Penny, his beloved co-star and owner, Bolt must harness all his \"super powers\" to find a way home."
        },
        {
            'name': 'Pirates of the Caribbean: On Stranger Tides',
            'poster_path': '/wNUDAq5OUMOtxMlz64YaCp7gZma.jpg',
            'genres': 'Adventure|Action|Fantasy',
            'overview': "Captain Jack Sparrow crosses paths with a woman from his past, and he's not sure if it's love -- or if she's a ruthless con artist who's using him to find the fabled Fountain of Youth. When she forces him aboard the Queen Anne's Revenge, the ship of the formidable pirate Blackbeard, Jack finds himself on an unexpected adventure in which he doesn't know who to fear more: Blackbeard or the woman from his past."
        },
    ]
};

const recommendationUrl = "";

function sendMovieRecommendationRequest() {
    console.log('hit search button')
    let movieInput = "";
    let userId = "";
    //TODO: grab text from html input element and store into movieInput object

    const input = document.getElementById("movie-input")
    const inputValue = input.value;
    console.log(inputValue);
    
    const searchStatement = document.getElementById("search-statement");
    searchStatement.style.display = 'block';
    searchStatement.textContent = 'Searching movie recommendations for: "' + inputValue + '"';

    // TODO: Send the API request . http://localhost:8080/api/movie-recommendations?movie=Toy%20Story&userId=2
    // d3.json(recommendationUrl + "?movie=" + movieInput + "&userId=" + userId).then(loaddata);

    input.value = "";

}

document.getElementById("search-button").onclick = sendMovieRecommendationRequest;

var userinputElement = document.getElementById("movie-input");
userinputElement.addEventListener("keydown", function (e) {
    if (e.code === "Enter") {  //checks whether the pressed key is "Enter"
        sendMovieRecommendationRequest();
    }
});


loaddata(dummyData);

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
            <h2>${recommendation.name}</h2>
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
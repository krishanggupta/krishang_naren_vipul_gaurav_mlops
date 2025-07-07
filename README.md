# krishang_naren_vipul_gaurav_mlops

Link to Docker Image:
		
		URL: https://hub.docker.com/r/naren82/team1

	1) Command to Pull the docker image:

			docker pull naren82/team1:latest

	2) Command to run the docker container:
	
			docker run -it -p 5004:5004 naren82/team1


Flask APIs:

	1) POST Method --> To Train the model
		
			URL: http://127.0.0.1:5004/sentiment-analysis/train
			Sample Body:
						{
							"C":[1,0.1,0.01],
							"penalty":["l2"],
							"solver":["liblinear","lbfgs"]
						}
						
	2) POST Method --> To Predict based on a data point
	
			URL: http://127.0.0.1:5004/sentiment-analysis/predict
			Sample Body:
								
					{
						"text":"Movie is super, awesome direction. we should watch the movie"
					}
					
	3) GET Method --> To retrieve the best parameters for prediction:
			
			URL: http://127.0.0.1:5004/sentiment-analysis/get-best-parameter
			
			No Body required.
			
			Sample Response:
					
					{"best parameters": {"C": 1, "penalty": "l1", "solver": "liblinear"}}
					
	

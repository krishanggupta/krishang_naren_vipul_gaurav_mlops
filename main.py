from flask import Flask, request
import sentiment_analyzer

app = Flask("__name__")

@app.get("/")
def ping():
    return "This site is up !!!"

@app.post("/sentiment-analysis/train")
def train_model():
    params = request.get_json()
    return sentiment_analyzer.train_model(params)

@app.post("/sentiment-analysis/predict")
def predict_sentiment():
    params = request.get_json()
    result  = sentiment_analyzer.predict(params)
    return result

@app.get("/sentiment-analysis/get-best-parameter")
def get_best_parameter():
    return sentiment_analyzer.get_best_parameter()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
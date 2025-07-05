from flask import Flask, request
import model

app = Flask("__name__")

@app.get("/")
def ping():
    return "This site is up !!!"

@app.post("/sentiment-analysis/train")
def train_model():
    params = request.get_json()
    return params

@app.post("/sentiment-analysis/predict")
def predict_sentiment():
    params = request.get_json()
    result  = model.predict(params)
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
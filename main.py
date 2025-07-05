from flask import Flask, request

app = Flask("__name__")

@app.get("/")
def ping():
    return "This site is up !!!"

@app.post("/sentiment-analysis/train")
def train_model():
    params = request.get_json()
    return params

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
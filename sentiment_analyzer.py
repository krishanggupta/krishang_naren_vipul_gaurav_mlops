import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from nltk.corpus import stopwords
import json, numpy as np
import pickle, jsonify


def load_data():
    # Load data
    train_df = pd.read_csv(r'data\train_dataset.csv')  # Columns: 'review', 'sentiment'
    test_df = pd.read_csv(r'data\test_dataset.csv')    # Same columns
    return (train_df, test_df)

def train_classifier(param_grid):
    # Download NLTK stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    train_df, test_df = load_data()
    # Preprocess reviews
    train_df['clean_review'] = train_df['review'].apply(preprocess_text, stop_words=stop_words)
    test_df['clean_review'] = test_df['review'].apply(preprocess_text, stop_words=stop_words)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train_df['clean_review'])
    X_test = tfidf.transform(test_df['clean_review'])

    # Encode sentiment (assuming 'positive' and 'negative')
    y_train = train_df['sentiment'] 
    y_test = test_df['sentiment']

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    cf = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    grid = train_with_tuning(model, param_grid, X_train, y_train)

    # Print the best parameters and best score
    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    cf = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    serialize_model(best_model, tfidf, grid)
        
    return report
    
def train_with_tuning(classifier, param_grid, X_train, y_train):
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #           'penalty': ['l1', 'l2'],
    #           'solver': ['liblinear', 'lbfgs']}

    # Perform grid search with cross-validation
    grid = GridSearchCV(classifier, param_grid, cv=5, refit=True, verbose=0)

    # Fit the grid search to the training data
    grid.fit(X_train, y_train)

    return grid

    
def train_model(params):
    cf = train_classifier(params)
    return json.dumps({"Result":cf})


# Preprocessing function
def preprocess_text(text, stop_words):

    text = text.lower()                              # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]      # Remove stopwords
    return ' '.join(tokens)

def serialize_model(tfidf_vectorizer, model, gridmodel):
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    with open("sentiment_analysis_best_model.pkl","wb") as sentiment_analysis_best_model:
        pickle.dump(model,sentiment_analysis_best_model)

    with open("grid_model.pkl","wb") as grid_model:
        pickle.dump(gridmodel,grid_model)

def deserialize_model():
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer:
        vectorizer = pickle.load(tfidf_vectorizer)

    with open("sentiment_analysis_best_model.pkl", "rb") as sentiment_analysis_best_model:
        best_model = pickle.load(sentiment_analysis_best_model)

    with open("grid_model.pkl","rb") as grid_model:
        grid = pickle.load(grid_model)
        
    return (vectorizer,best_model, grid)

def predict(params):
    model, vectorizer, grid =  deserialize_model()
    text_to_predict = params["text"]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    text_to_predict = preprocess_text(text_to_predict, stop_words)

    # TF-IDF Vectorization
    string_list = [text_to_predict]
    to_predict = vectorizer.transform(string_list)
    y_pred = model.predict(to_predict)
    sentiment =  "Positive Sentiment" if  y_pred[0] else "Negative sentiment"
    # print(accuracy)
    return json.dumps({"Result":sentiment})

import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from nltk.corpus import stopwords
import json, numpy as np



def load_data():
    # Load data
    train_df = pd.read_csv(r'data\train_dataset.csv')  # Columns: 'review', 'sentiment'
    test_df = pd.read_csv(r'data\test_dataset.csv')    # Same columns
    return (train_df, test_df)


def train_classifier(param_grid):
    # Download NLTK stopwords
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))

    train_df, test_df = load_data()
    # Preprocess reviews
    train_df['clean_review'] = train_df['review'].apply(preprocess_text)
    test_df['clean_review'] = test_df['review'].apply(preprocess_text)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train_df['clean_review'])
    X_test = tfidf.transform(test_df['clean_review'])

    # Encode sentiment (assuming 'positive' and 'negative')
    y_train = train_df['sentiment'] #//.map({1: 1, 0: 0})
    y_test = test_df['sentiment'] #.map({'positive': 1, 'negative': 0})

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    cf = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    cf = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return cf
    

    
def train_model(params):

    cf = train_classifier(params)

    return json.dumps(cf.tolist())


# Preprocessing function
def preprocess_text(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that's", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    text = text.lower()                              # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]      # Remove stopwords
    return ' '.join(tokens)

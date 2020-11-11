from flask import Flask, request, render_template
from pickle import load
import pandas as pd

from bs4 import BeautifulSoup
import re
#from nltk.corpus import stopwords

model = load(open('model.pkl', 'rb'))
vectorizer = load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')


def preprocessing(tweet):
    text = BeautifulSoup(tweet,).get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    #stopword = set(stopwords.words("english"))
    #text = [w for w in words if not w in words]
    
    return(" ".join(text))

def clean_t(df):
    nb_tweets = df["text"].size
    clean_tweets = []
    for i in range(0, nb_tweets):                                                                
        clean_tweets.append(preprocessing(df["text"][i]))
        
    return clean_tweets

def predictions(rf_model, X_test):
    rf_predictions = rf_model.predict(X_test)
    
    return rf_predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        text = request.form['text']

        input_variables = pd.DataFrame(columns = ['text'])
        input_variables.loc[0, 'text'] = text

        clean_tweets = clean_t(input_variables)

        test_features = vectorizer.transform(clean_tweets)
        test_features = test_features.toarray()

        prediction = predictions(model, test_features)[0]

        return render_template('index.html', text = text, prediction = prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import model_utils
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model

mutils = model_utils.modelUtils()


# model = pickle.load(open('models/news_topic_classification_RNN.pkl', 'rb'))
# tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
model = load_model('./models/model.h5')
with open('./models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

TOPIC_DICT = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
MAX_SEQUENCES_LENGTH = 20
BATCH_SIZE = 128


app = Flask(__name__)

@app.get('/')
def home():
    return render_template('home.html')

@app.get('/predict')
@app.post('/predict')
def predict():
    title = request.form['title']

    cleaned_titles = [mutils.dataPreprocessing(title)]
    text_sequences = tokenizer.texts_to_sequences(cleaned_titles)
    text_inputs = preprocessing.sequence.pad_sequences(text_sequences, maxlen=MAX_SEQUENCES_LENGTH, padding='post')

    predictions = model.predict(text_inputs, batch_size=BATCH_SIZE)

    value = predictions[0]

    res = TOPIC_DICT[np.where(value == max(value))[0][0]]
    max_percent = float(np.max(value))
    return {
        'result': res,
        'percentage': max_percent
    }
    


# @app.get('/')
# def home():
#     return '<h1>Hello World</h1>'

# @app.route('/test')
# def test():
#     return '<h2>test</h2>'

if __name__ == '__main__':
    app.run(debug=True)
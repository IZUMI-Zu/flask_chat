"""
This module implements a Flask server with three endpoints for different tasks: sentiment prediction, text similarity,
and chat-based responses.

The sentiment prediction endpoint takes a POST request with JSON data containing raw text. It processes the text,
performs sentiment prediction using a trained model, and returns the sentiment and the model's score.

The text similarity endpoint takes a POST request with JSON data containing two pieces of text. It computes the cosine
similarity between the texts using a pre-trained Word2Vec model and returns the similarity score and a quantized
similarity level.

The chat endpoint takes a POST request with JSON data containing a text message. It uses a pre-trained chat model to
generate a response to the message and returns the generated response.

The module also includes utility functions for text preprocessing and vectorization.

Note:
- Before running the server, make sure to set the appropriate values for `MAX_SEQ_LENGTH`, `MODEL_NAME`,
  and `MODEL_PATH_PREFIX`.
- The module assumes the existence of external functions `select_chinese()`, `pro_sentence()`, `get_index()`,
  `judge()`, `split_sentence()`, `wordaver()`, and `load_word2vec_model()`. Make sure they are defined and accessible.
"""

import re
from flask import Flask, request, jsonify
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from chat_box.model_predictor import ModelPredictor

from utils.utils import select_chinese, pro_sentence, get_index, judge, \
    split_sentence, wordaver, load_word2vec_model

app = Flask(__name__)

MAX_SEQ_LENGTH = 30 # will set this manually for demo
MODEL_NAME = 'first_model.h5' # will set this manually for demo
MODEL_PATH_PREFIX = './model/' # will set this manually for demo

@app.route('/api/predict', methods=['POST'])
def predict():
    """ 
    This function is a Flask server endpoint that takes POST requests 
    with JSON data containing raw text. It processes the text, performs
    sentiment prediction on it using a trained model, and returns the 
    sentiment and the model's score.

    The request data should come in a JSON format with a 'text_ori' field
    storing the raw text for prediction.

    Returns: 
    A dictionary in JSON format containing:
    - 'sentiment': a string describing the sentiment ('positive' or 'negative') 
      of the input text according to the trained model.
    - 'score': string of model's output score.
    """
    mod = load_model(MODEL_PATH_PREFIX + MODEL_NAME)

    data = request.get_json(force=True) # get data from POST request
    text_ori = data['text_ori']

    text_pre1 = select_chinese(text_ori)
    text_pre2 = pro_sentence(text_pre1)
    stop = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    text = re.sub(stop, '', text_pre2).split()
    sentence1 = []
    sentence1.append(text)
    testdata = list(map(get_index, sentence1))
    testdata_fin = pad_sequences(testdata, maxlen=MAX_SEQ_LENGTH)

    scores = mod.predict(testdata_fin)
    pred_score = judge(scores)

    if pred_score == 1:
        result = {"sentiment": "positive", "score": str(scores)}
    else:
        result = {"sentiment": "negative", "score": str(scores)}

    return jsonify(result)

@app.route('/api/similarity', methods=['POST'])
def similarity():
    """
    The function is a Flask route handler that takes a POST request containing two pieces of text and
    returns a JSON object containing the cosine similarity between them (as computed using a pre-trained 
    Word2Vec model) and a quantized similarity score ('low', 'middle', 'high').

    The POST request should include a JSON object with 's1' and 's2' keys, each corresponding to a piece
    of text. The function pre-processes and vectorizes these texts, computes their cosine similarity, and 
    uses this similarity to determine whether the similarity is 'low' (score <= 0.6), 'middle' 
    (0.6 < score <= 0.75), or 'high' (score > 0.75).

    Returns:
    - A JSON object with "similarity" set to "low", "middle", or "high" depending on the computed cosine
      similarity and "score" set to the actual cosine similarity between the vectors of 's1' and 's2'.

    Note:
    - It employs several externally defined functions: `load_word2vec_model()`, `pro_sentence()`, 
      `select_chinese()`, `split_sentence()` and `wordaver()`. Make sure they are defined and accessible 
      before calling `similarity()`.
    - It assumes that the JSON payload of the POST request contains 's1' and 's2' keys. If not, it may 
      throw a KeyError.
    """
    model = load_word2vec_model()

    data = request.get_json(force=True) # get data from POST request
    text1 = data['s1']
    text1_pre = pro_sentence(select_chinese(text1))
    text1 = split_sentence(text1_pre)

    text2 = data['s2']
    text2_pre = pro_sentence(select_chinese(text2))
    text2 = split_sentence(text2_pre)

    s1_aver = wordaver(model, text1) #s1的向量表示
    s2_aver = wordaver(model, text2) #s2的向量表示

    scores = cosine_similarity(s1_aver.reshape(1,-1),s2_aver.reshape(1,-1))

    if scores <= 0.6:
        result = {"similarity": "low", "score": str(scores)}
    elif scores <= 0.75:
        result = {"similarity": "middle", "score": str(scores)}
    else:
        result = {"similarity": "high", "score": str(scores)}

    return jsonify(result)  

@app.route('/api/chat', methods=['POST'])
def chat():

    predictor = ModelPredictor('model/model_data', "model/inp.vocab", "model/tar.vocab")

    data = request.get_json(force=True) # get data from POST request
    text = data['text']
    
    result = {"text": str(predictor.predict(text))}
    return jsonify(result)


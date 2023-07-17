"""
This module contains functions for text preprocessing, specifically for sentiment analysis tasks.

The functions perform tasks such as:
- Reading text lines from a string or a txt file.
- Selecting Chinese characters from a text string.
- Processing (segmenting and stop-word removal) a sentence in Chinese.
- Removing specified punctuation from a sentence and splitting it into words.
- Judging the sentiment (positive or negative) based on a given score.

The module is dependent on the 're' and 'jieba' packages for regular
expression and Chinese word segmentation, respectively. It also makes use of an 
external stop-words file ('hit_stopwords.txt') for the processing of Chinese sentences.

Date: 2023-07-017

Note: Ensure the provided scores for sentiment judgement are probabilities (between 0 and 1).
Higher scores indicate more positive sentiment.
"""

import re
import jieba
import numpy as np
from gensim.models import word2vec

STOPWORDS_FILE = './static/hit_stopwords.txt' # path to stopwords file
WORD_2_VEC_MODEL = './model/worddd.model' # path to word2vec model

def read_txt(string: str):
    """ 
    This function reads a text string, each line contained as an element in a list.

    Arguments:
    - string: String data to be read. Each line in the string will be returned as an
    individual element.

    Returns: A list where each line in the input string is an individual element.
    """
    # Remove specified characters at the head and tail of the string
    txt = [line.strip() for line in string]
    return txt

def read_txt_from_file(filepath):
    """ 
    This function reads a text file, each line contained as an element in a list.

    Arguments:
    - string: String data to be read. Each line in the string will be returned as an
    individual element.

    Returns: A list where each line in the input string is an individual element.
    """
    file = open(filepath,'r',encoding='utf-8')
    # Remove specified characters at the head and tail of the string
    txt = [line.strip()for line in file]
    file.close()
    return txt

def select_chinese(text):
    """
    This function selects and returns only Chinese characters from the provided text.

    Arguments:
    - text: String input from which Chinese characters will be selected.

    Returns: A string that contains only the Chinese characters found in the input text.
    """
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chine_select_text = re.sub(pattern, '', str(text)).strip()
    return chine_select_text

def pro_sentence(sentence):
    """
    This function segments Chinese text and removes stop-words.

    Arguments:
    - sentence: The Chinese sentence string that will be segmented and processed.

    Returns: A processed string whose words are segmented and stop-words are removed.
    
    Note: This function relies on an external file 'hit_stopwords.txt' for stop-words.
    """
    # Chinese word segmentation
    sentence_segmentation = jieba.cut(sentence.strip())
    # Read the stopwords file
    stopwords = read_txt_from_file(STOPWORDS_FILE)
    protxt = ""
    for word in sentence_segmentation:
        if word not in stopwords:
            if word != '/t':
                protxt += word
                protxt += " "
    return protxt

def split_sentence(sentence):
    """
    This function removes all specified punctuation from a sentence and splits it into words.

    Arguments:
    - sentence: The string input which will be processed.

    Returns: A list of words from the input sentence, after removing specified
    punctuation and splitting on whitespace.

    Note: The function uses regular expressions for removing punctuation. The present configuration 
    removes the following punctuation: [!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]
    """
    stop ='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(stop,'',sentence)
    return sentence.split()

def judge(score):
    """
    This function assigns a sentiment label based on a given score.

    Given a score between 0 and 1, the function returns 1 (positive sentiment)
    if the score exceeds a threshold of 0.5, otherwise it returns 0 (negative sentiment).

    Arguments:
    - score: Float in the range [0, 1] which is the output probability of a model.

    Returns: 1 if score > 0.5; 0 otherwise.

    Note: 
    - Make sure the score parameter is indeed a probability (between 0 and 1).
    - Higher scores indicate more positive sentiment.
    """
    return 1 if score > 0.5 else 0

def get_index(sentence):
    """ 
    This function converts words from a sentence into corresponding indices, 
    using a word2vec model's vocabulary.

    Given a sentence, the function converts the sentence into a sequence of 
    indices. If a word in the sentence is not in the model's vocabulary,
    it is skipped.

    Args:
    - sentence: A list of words that forms the sentence.

    Returns: A list of integers, which are the indices from the model's 
    vocabulary that correspond to the words in the input sentence.

    Note: 
    - The function depends on the gensim.models.Word2Vec model.
    - Words not found in the model's vocabulary will be ignored.
    """
    model = load_word2vec_model()
    vocab_list = list(model.wv.index_to_key)
    word_index = {word:index for index, word in enumerate(vocab_list)}
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

def wordaver(model, words):
    """
    This function calculates the average vector for a list of words.

    Given a list of words and a trained word2vec model, the function computes the 
    average vector by taking the mean of the vectors of the words in the list.

    Args:
    - model: A trained word2vec model.
    - words: A list of words.

    Returns: A numpy array representing the mean vector of the input words' vectors.

    Note: 
    - The function depends on the gensim.models.Word2Vec model.
    - The words should be in the model's vocabulary.
    """
    return np.mean([model.wv.get_vector(word) for word in words],axis=0)


def mark_score(score):
    """
    This function categorizes a given score into one of three categories.

    The function takes a score (presumably a similarity score between two texts) 
    and marks it as 'high', 'medium', or 'low' based on its value.

    Args: 
    - score: A float that represents a similarity score.

    Prints:
    - "文本相似度高" if the score is greater than  0.75.
    - "文本相似度中" if the score is between 0.6 and 0.75.
    - "文本相似度低" if the score is less than or equal to 0.6.

    """
    if score <= 0.6:
        print("文本相似度低")
    elif score <= 0.75:
        print("文本相似度中")
    else:
        print("文本相似度高")

def load_word2vec_model(model_path=WORD_2_VEC_MODEL):
    """
    This function is used to load a pre-trained Word2Vec model from a given path.

    Given a path to a saved Word2Vec model, it uses the gensim.models.Word2Vec.load() function 
    to load the model and return it.

    Args:
    - model_path (str): The path to the saved Word2Vec model. By default, it is set to WORD_2_VEC_MODEL
                        which is presumably a constant defined elsewhere in the code.

    Returns: A gensim.models.Word2Vec instance representing the loaded model.

    Note:
    - The function depends on the gensim.models.Word2Vec model.
    - The model at the specified path should be a saved Word2Vec model.
    """
    return word2vec.Word2Vec.load(model_path)


    
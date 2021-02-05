import collections
import json
import pickle
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def process_query(query):
    test_path = "../data/test.pkl"
    vectorizer_path = "../data/corpus_vectorizer.pkl"
    corpus_tfidf_path = "../data/corpus_tfidf.pkl"

    test_data = pickle.load(open(test_path, "rb"))
    print(test_data)

    corpus_vectorizer = pickle.load(open(vectorizer_path, "rb"))
    print(corpus_vectorizer)

#Function to tokenize the text blob
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemma = find_lemma(tokens)
    return lemma

# Lemmatize words for better matching
def find_lemma(tokens):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    result = []
    for word in tokens:
        lemma_word = wordnet_lemmatizer.lemmatize(word)
        result.append(lemma_word)
    return result

# Call main method
if __name__ == "__main__":
    process_query("Yolo")

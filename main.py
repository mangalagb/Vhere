# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_render_template]
import nltk
from flask import Flask, request, render_template
import vhere.code.Recommender as reco

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html", title="HOME PAGE")


@app.route('/results', methods=['POST'])
def results():
    default_query = request.form['text']
    return render_template('result.html', recos=reco.recommend(default_query).items())

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


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8085, debug=True)
# [END gae_python38_render_template]

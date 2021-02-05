import collections
import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def process_query(query):
    # Paths to load the trained model
    base_path = Path(__file__).parent.parent
    corpus_dict_path = base_path / "data/raw_data/summaries/summary_file.txt"
    vectorizer_path = base_path / "data/trained_model/corpus_vectorizer.pkl"
    corpus_tfidf_path = base_path / "data/trained_model/corpus_tfidf.pkl"

    print(vectorizer_path)
    corpus_vectorizer = pickle.load(open(vectorizer_path, "rb"))
    print("ui")
    corpus_tfidf = pickle.load(open(corpus_tfidf_path, "rb"))
    corpus_dict = collections.OrderedDict()

    with open(corpus_dict_path, 'r') as read_file:
        corpus_dict = json.load(read_file)

    tfidf_matrix_test = corpus_vectorizer.transform([query])
    cosine_similarity_matrix = cosine_similarity(corpus_tfidf, tfidf_matrix_test)
    return corpus_dict, cosine_similarity_matrix


# Map the original corpus to its cosine score
def get_recommendations(corpus_dict, cosine_similarity_matrix):
    items = list(corpus_dict.items())
    recommendation_dict = collections.OrderedDict()

    for i in range(0, len(items)):
        corpus_text = items[i]
        title = corpus_text[0]
        cosine_score = cosine_similarity_matrix[i]
        recommendation_dict[title] = cosine_score

    sorted_recommendation_dict = {k: v for k, v in
                                  sorted(recommendation_dict.items(), reverse=True, key=lambda item: item[1])}
    return sorted_recommendation_dict


# Print the recommendations
def print_recommendations(corpus_dict, sorted_recommendation_dict, presult=False):
    print("Based on your search query, look at these datasets from CSA :")

    # We limit the search results to 10
    limit = 10
    count = 0
    result = collections.OrderedDict()
    for title, cosine_similarity in sorted_recommendation_dict.items():
        if cosine_similarity == 0.0 or count == limit:
            break
        result[title] = corpus_dict[title]
        count += 1

    if presult:
        for k, v in result.items():
            print(k)
            print(v)
            print("-----------------------------------------------------")
    return result


# # Function to tokenize the text blob
# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemma = find_lemma(tokens)
#     return lemma
#
#
# # Lemmatize words for better matching
# def find_lemma(tokens):
#     wordnet_lemmatizer = nltk.WordNetLemmatizer()
#     result = []
#     for word in tokens:
#         lemma_word = wordnet_lemmatizer.lemmatize(word)
#         result.append(lemma_word)
#     return result

def get_user_query(query_dict):
    # Read the input document that needs to be compared
    return query_dict["query"]


def recommend(query):
    qdict = {"query": query}
    user_query = get_user_query(qdict)
    corpus_dict, cosine_similarity_matrix = process_query(user_query)
    recommendation_dict = get_recommendations(corpus_dict, cosine_similarity_matrix)
    return print_recommendations(corpus_dict, recommendation_dict)

def main():
    query_dict = {"query": "I want to know more about WINDII and Doppler and performance"}

    user_query = get_user_query(query_dict)
    corpus_dict, cosine_similarity_matrix = process_query(user_query)
    recommendation_dict = get_recommendations(corpus_dict, cosine_similarity_matrix)
    print_recommendations(corpus_dict, recommendation_dict, True)

# Call main method
if __name__ == "__main__":
    main()

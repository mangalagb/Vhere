import pickle

def main():
    corpus_dict_path = "../data/test.pkl"

    with open(corpus_dict_path, 'rb') as handle:
        b = pickle.load(handle)
        print(b)


if __name__ == "__main__":
    main()

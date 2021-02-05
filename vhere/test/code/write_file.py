import pickle
from pathlib import Path

def main():
    a = {'hello': 'world'}

    # base_path = Path(__file__).parent.parent
    # print(base_path)
    corpus_dict_path = "../data/test.pkl"

    with open(corpus_dict_path, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
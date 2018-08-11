import os
import re
import math
import time
import numpy as np
import snowballstemmer as stem
from scipy.sparse import csr_matrix, linalg
import sklearn.preprocessing


def gen_words(text, stemming=stem.stemmer('english')):
    """Create generator.
    :param text: some string
    :param stemming: variant of stemming algorithm
    :return: generator giving stemmed words from text
    """
    for word in stemming.stemWords(re.findall(r"[\w']+", text.lower())):
        yield word


def parse_file(file_path, bow, stemming):
    """Create bow set by parsing file.
    :param file_path: path to file
    :param bow: bag_of_words set
    :param stemming: variant of stemming algorithm
    :return: bag_of_words set for parsed file
    """
    terms = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            for word in gen_words(line.lower(), stemming):
                if word not in bow:
                    bow[word] = len(bow)

                if bow[word] in terms:
                    terms[bow[word]] += 1
                else:
                    terms[bow[word]] = 1

    return terms


def idf_matrix(matrix, bow):
    """Use inverse document frequency to reduce meaning of frequent words.
    :param bow: bag_of_words set
    :param matrix: term_by_document matrix
    :return: none
    """
    for word in bow:
        docs = sum(1 for doc in matrix if bow[word] in matrix[doc])
        idf = math.log10(len(matrix) / docs)
        for doc in matrix:
            if bow[word] in matrix[doc]:
                matrix[doc][bow[word]] *= idf


def create_matrix(dir_path, svd=False, k=30, idf=True):
    """Create bag_of_words and term_by_document matrix.
    :param idf: whether to apply idf for each word
    :param k: parameter for svd
    :param svd: whether to apply low rank approximation
    :param dir_path: path to directory with documents
    :returns: bag_of_words, documents names, term_by_document matrix
    """
    bow = {}
    stemming = stem.stemmer('english')
    docs_terms = {name: parse_file(dir_path + "/" + name, bow, stemming)
                  for name in os.listdir(dir_path)}

    if idf:
        idf_matrix(docs_terms, bow)

    rows = []
    cols = []
    data = []
    docs = list(docs_terms.keys())
    for i, doc in enumerate(docs):
        col = docs_terms[doc].keys()
        data.extend([docs_terms[doc][x] for x in col])
        cols.extend(col)
        rows.extend([i] * len(col))

    matrix = csr_matrix((data, (rows, cols)),
                        shape=(len(docs_terms), len(bow)), dtype=float)

    if svd:
        u, s, v = linalg.svds(matrix, k)
        matrix = csr_matrix(u.dot(np.diag(s)).dot(v))

    sklearn.preprocessing.normalize(matrix, norm='l1', axis=1, copy=False)

    return bow, docs, matrix


def create_vec(text, bow):
    """Change user query into bag_of_words vector.
    :param text: user query
    :param bow: bag_of_words set
    :return: normalised bag_of_words vector
    """
    vec = np.zeros(shape=len(bow))
    for word in gen_words(text.lower()):
        if word in bow:
            vec[bow[word]] += 1
    return vec / np.linalg.norm(vec)


def fix_arr(max_values, indexes, n):
    """Sort in reverse order max_values.
    :param max_values: arr of max_values
    :param indexes: arr of indexes
    :param n: len of indexes/max_values
    :return: none
    """
    i = n - 1
    while i > 0 and max_values[i] > max_values[i - 1]:
        max_values[i], max_values[i - 1] = max_values[i - 1], max_values[i]
        indexes[i], indexes[i - 1] = indexes[i - 1], indexes[i]
        i -= 1


def query(matrix, vec, n=1):
    """Find n documents with highest correlation to query.
    :param matrix: term_by_document matrix
    :param vec: query from user
    :param n: number of documents to find
    :return: n max files indexes with highest correlation
    """
    arr = matrix.dot(vec)
    max_values = np.zeros(shape=n)
    indexes = np.zeros(shape=n, dtype=int)

    for i, elem in enumerate(arr):
        if elem > max_values[-1]:
            max_values[-1] = elem
            indexes[-1] = i
            fix_arr(max_values, indexes, n)
    return indexes


def display_results(docs, indexes):
    """Display results of query.
    :param docs: arr of names of files
    :param indexes: indexes of docs with highest correlation
    :return: none
    """
    print("\nTop %d results are:" % len(indexes))
    for index in indexes:
        print(docs[index])


def main(n=10, k=50, svd=True, idf=True, path='wiki'):

    bow, docs, matrix = create_matrix(path, svd, k, idf)

    text = input('\nEnter sentence: ')
    while text != 'quit':
        start_time = time.time()
        vec = create_vec(text, bow)
        display_results(docs, query(matrix, vec, n))
        end_time = time.time()
        print("time: " + str(end_time - start_time))
        text = input('\nEnter sentence: ')


if __name__ == "__main__":
    main()


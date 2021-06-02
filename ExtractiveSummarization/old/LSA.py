from nltk.stem import PorterStemmer
import numpy
from numpy.linalg import svd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import math

class LSA:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def create_dictionary(self, text):
        """
        :param text: Source text
        :return dict: dictionary where the key is a word and the value is its corresponding
                      row number in our word-sentence matrix (used in def create_matrix)
        """
        words = word_tokenize(text.lower())
        vocabulary = set() # set with unique stemmed words
        for word in words:
            word = self.stemmer.stem(word)
            if word in self.stop_words:
                continue
            vocabulary.add(word)
        return dict((w, i) for i, w in enumerate(vocabulary))


    def create_matrix(self, text, dictionary):
        """
        Creates a matrix of size |unique words|x|sentences|,
        where element (ij) - number of word i's occurences in sentence j.
        :param text: Source text
        :param dictionary: dict (word, word row number)
        :return matrix: Matrix of size |unique words|x|sentences|,
                        where element (ij) = 0 if sentence j does not contain
                        word i, otherwise = frequency of word i in source text.
        """
        sentences = sent_tokenize(text)

        words_count = len(dictionary)
        sentences_count = len(sentences)

        matrix = numpy.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                word = self.stemmer.stem(word)
                if word in dictionary:  # Stop words don't count in matrix
                    row = dictionary[word]
                    matrix[row, col] += 1
        rows, cols = matrix.shape
        if rows and cols:
            word_count = numpy.sum(matrix)  # Number of words in text
            for row in range(rows):
                unique_word_count = numpy.sum(matrix[row, :]) # word frequency
                for col in range(cols):
                    if matrix[row, col]:
                        matrix[row, col] = unique_word_count/word_count
        else:
            matrix = numpy.zeros((1, 1))
        return matrix


    def compute_ranks(self, matrix, n):
        """
        :param matrix: Word-sentence matrix with word frequency values.
        :param n: Number of sentences in the summary.
        :return list: List with ranks of sentences.
        """
        u_m, sigma, v_m = svd(matrix, full_matrices=False)
        powered_sigma = tuple(s**2 if i < n else 0.0 for i, s in enumerate(sigma))
        ranks = []
        # Iteration by columns of matrix (i.e. rows of its transpose)
        for column_vector in v_m.T:
            # Use formula from chapter 3.4
            rank = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))
        return ranks


    def summarize(self, text, n):
        """
        Generates summary using LSA
        :param text: Source text
        :param n: Number of sentences in the summary
        :return string: Summary
        """
        sentences = sent_tokenize(text)
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % (n, len(sentences)))
        d = self.create_dictionary(text)
        m = self.create_matrix(text, d)
        r = self.compute_ranks(m, n)
        rank_sort = sorted(((i, r[i], s) for i, s in enumerate(sentences)), \
                    key=lambda x: r[x[0]], reverse=True)
        top_n = sorted(rank_sort[:n])
        return ' '.join(x[2] for x in top_n)


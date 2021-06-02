from .BaseSummarizer import BaseSummarizer
from nltk.tokenize import sent_tokenize
import numpy
from numpy.linalg import svd
import math

class LSA(BaseSummarizer):

    def create_dictionary(self, text):
        """
        :param text: Source text
        :return dict: dictionary where the key is a word and the value is its corresponding
                      row number in our word-sentence matrix (used in def create_matrix)
        """
        words = self.word_tokenize_preprocessed(text)
        vocabulary = set(words) # set with unique stemmed words
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
            for word in self.word_tokenize_preprocessed(sentence):
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
        self.validate_summary_length(sentences, n)
        preprText = self.preprocess_document(text)
        d = self.create_dictionary(preprText)
        m = self.create_matrix(preprText, d)
        r = self.compute_ranks(m, n)
        return self.summary_from_sentence_scores(sentences, r, n)


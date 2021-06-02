from .BaseSummarizer import BaseSummarizer
from nltk.tokenize import sent_tokenize
import math
import numpy as np

class TF_IDF(BaseSummarizer):
    
    def __init__(self, documents=[]):
        """
        :param documents: Collection of documents that will be used to generate idf_table
        """
        super().__init__()
        self.create_dpw_table(documents)


    def create_dpw_table(self, documents=[]):
        """
        Builds a new doc_per_word_table using a collection of documents
        :param documents: collection of documents
        """
        # Dict - key: word, value: number of documents where the word occurs
        self.doc_per_word_table = dict()
        self.total_documents = 0
        self.update_dpw_table(documents)


    def update_dpw_table(self, documents=[]):
        """
        Updates doc_per_word_table by processing
        an additional collection of documents
        :param documents: additional collection of documents
        """
        for doc in documents:
            doc = self.preprocess_document(doc)
            # converting list to set will delete any duplicate words
            doc_words = set(self.word_tokenize_preprocessed(doc))

            for word in doc_words:
                if word in self.doc_per_word_table:
                    self.doc_per_word_table[word] += 1
                else:
                    self.doc_per_word_table[word] = 1

        self.total_documents += len(documents)


    def create_tf_table(self, text) -> dict:
        """
        Returns table with unique words and their TF values
        TF = number of occurences of a word / total number of words in a document
        :param text: Source text
        """
        words = self.word_tokenize_preprocessed(text)
        freqTable = dict()
        tfTable = dict()

        totalWords = len(words)
        for word in words:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        
        for word in freqTable:
            tfTable[word] = freqTable[word] / float(totalWords)

        return tfTable


    def create_tf_idf_table(self, text):
        """
        Returns table with unique words and their TF-IDF values given collection of documents
        TF-IDF = TF * IDF
        TF = number of occurences of a word / total number of words
        IDF = log(total number of documents / number of documents that contain the word)
        :param text: Source text
        """
        tf_idf_table = dict()
        tf_table = self.create_tf_table(text)

        for word in tf_table:
            if word in self.doc_per_word_table:
                idf = math.log2(self.total_documents / self.doc_per_word_table[word])
                tf_idf_table[word] = tf_table[word] * idf
            else:
                raise KeyError("Word '%s' was not found in idf_table." % word)

        return tf_idf_table


    def score_sentences(self, sentences, tfIdfTable) -> dict:
        """
        Calculates scores for sentences based on word frequency
        :param sentences: list of tokenized sentences
        :param freqTable: table of words and their frequencies
        :return sentenceScores: list of sentence scores
        """
        sentenceScores = []
        
        for sentence in sentences:
            words = set(self.word_tokenize_preprocessed(sentence))
            sentScore = 0
            for word in words:
                if word in tfIdfTable:
                    sentScore += tfIdfTable[word]
            if len(words) > 0:
                sentScore /= len(words)
            sentenceScores.append(sentScore)

        return np.array(sentenceScores)


    def summarize(self, text, n):
        """
        Returns summary with n sentences using TF-IDF algorithm
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        self.validate_summary_length(sentences, n)
        preprText = self.preprocess_document(text)
        tfIdfTable = self.create_tf_idf_table(preprText)
        sentenceScores = self.score_sentences(sentences, tfIdfTable)

        return self.summary_from_sentence_scores(sentences, sentenceScores, n)


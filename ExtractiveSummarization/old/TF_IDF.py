import math
import glob
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

class TF_IDF:

    def preprocess_document(self, document):
        document = re.sub('[^A-Za-z .-]+', ' ', document)
        document = document.replace('-', '')
        document = document.replace('…', '')
        document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')
        return document

    def word_tokenize_preprocessed(self, text):
        words = word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words if word.isalnum() and \
                 word not in self.stop_words]
        # words = [self.stemmer.stem(word) for word in words]
        return words

    def pre_build_idf_table(self):
        """
        Builds idf_table using a collection of documents initialized in the constructor
        before summarizing any of the documents. Method is called in the constructor only
        if you specify the preBuildIdf parameter as True.
        """
        doc_per_word_table = dict() # in how many documents does a word occur
        
        for doc in self.documents:
            # converting list to set will delete any duplicate words
            doc = self.preprocess_document(doc)
            doc_words = set(self.word_tokenize_preprocessed(doc))

            for word in doc_words:
                if word in doc_per_word_table:
                    doc_per_word_table[word] += 1
                else:
                    doc_per_word_table[word] = 1

        total_documents = len(self.documents)
        idf_table = dict()

        for word in doc_per_word_table:
            idf_table[word] = math.log2(total_documents / float(doc_per_word_table[word]))

        return idf_table

    def __init__(self, documents, preBuildIdf = False):
        """
        :param documents: collection of documents
        :param preBuildIdf: if True, idf_table will be pre-built
                            (RECOMMENDATION: do it only if you plan to
                             summarize all documents in the collection)
        """
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        self.preBuildIdf = preBuildIdf
        self.documents = documents
        self.text_index = None

        if self.preBuildIdf:
            self.idf_table = self.pre_build_idf_table()
        else:
            self.idf_table = None


    def _create_tf_table(self, words) -> dict:
        """
        Returns table with unique words and their TF values
        TF = number of occurences of a word / total number of words in a document
        :param words: list of words in the main document
        """

        freqTable = dict()
        tfTable = dict()

        totalWords = len(words)
        for word in words:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        
        uniqueWords = set(words)
        for word in uniqueWords:
            tfTable[word] = freqTable[word] / float(totalWords)

        return tfTable


    def _create_idf_table(self, uniqueWords):
        """
        Returns table with unique words and their IDF values given collection of documents
        IDF = log(total number of documents / number of documents that contain the word)
        :param uniqueWords: list of unique words in the main document
        If the preBuildIdf parameter in the constructor is True, the method is not called.
        """
        doc_per_word_table = dict() # in how many documents does a word occur
        for word in uniqueWords:
            # not 0 because 'documents' list doesn't contain our main document
            doc_per_word_table[word] = 1
        
        for i in range(len(self.documents)):
            if self.text_index and i == self.text_index:  # if doc is main doc ignore
                continue
            doc = self.documents[i]
            doc_words = self.word_tokenize_preprocessed(doc)

            for word in uniqueWords:
                if word in doc_words:
                    doc_per_word_table[word] += 1

        total_documents = len(self.documents) + 1
        idf_table = dict()

        for word in uniqueWords:
            idf_table[word] = math.log2(total_documents / float(doc_per_word_table[word]))

        return idf_table



    def _create_tf_idf_table(self, words):
        """
        Returns table with unique words and their IDF values given collection of documents
        IDF = log(total number of documents / number of documents that contain the word)
        :param words: list of words in the document
        """
        tf_idf_table = dict()

        tf_table = self._create_tf_table(words)
        uniqueWords = set(words)
        # if idf_table was prebuilt there's no need to create it
        if not self.preBuildIdf:
            self.idf_table = self._create_idf_table(uniqueWords)

        for word in uniqueWords:
            if word in self.idf_table:
                tf_idf_table[word] = tf_table[word] * self.idf_table[word]
            else:
                raise KeyError("Word '%s' was not found in idf_table." % word)

        return tf_idf_table


    def _score_sentences(self, sentences, tfIdfTable) -> dict:
        """
        Calculates scores for sentences based on word frequency
        :param sentences: list of tokenized sentences
        :param freqTable: table of words and their frequencies
        :return sentenceScores: list of sentence scores
        """
        sentenceScores = []
        sentIndex = 0

        for sentence in sentences:
            words = set(self.word_tokenize_preprocessed(sentence))
            sentScore = 0
            for word in words:
                if word in tfIdfTable:
                    sentScore += tfIdfTable[word]

            if len(words) > 0:
                sentScore /= len(words)
            sentenceScores.append(sentScore)
            sentIndex += 1

        return sentenceScores

    def summarize(self, text, text_index, n):
        """
        Returns summary with n sentences
        :param text: Source text
        :param text_index: index of sorce text in list of documents
            if text_index is None: source text HAS been removed from documents; do nothing
            else: source text is in documents; document[text_index] will be ignored
                  when creating idf_table
            
            There should be no source text in documents since it's already provided as 'text'.
        :param n: number of sentences in summary
        """
        self.text_index = text_index
        sentences = sent_tokenize(text)
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % \
                             (n, len(sentences)))
        preprText = self.preprocess_document(text)
        words = self.word_tokenize_preprocessed(preprText)
        tfIdfTable = self._create_tf_idf_table(words)
        # print({k: v for k, v in sorted(freqTable.items(), key=lambda item: item[1], reverse=True)})
        sentenceScores = np.array(self._score_sentences(sentences, tfIdfTable))
        nBestIndexes = np.argpartition(sentenceScores, -n)[-n:] # indexes of sentences with n best scores
        nBestIndexes = sorted(nBestIndexes)

        summary = ''
        for index in nBestIndexes:
            summary += sentences[index] + " "

        self.text_index = None  # reset text_index once completed
        return summary[:-1]  # remove last space


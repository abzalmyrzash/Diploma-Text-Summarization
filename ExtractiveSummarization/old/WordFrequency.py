import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re

class WordFrequency:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def preprocess_document(self, document):
        document = re.sub('[^A-Za-z0-9]+', ' ', document)
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

    def _create_frequency_table(self, text) -> dict:
        """
        Returns table with words and their frequencies (number of word's occurences in source text)
        :param text: source text
        """

        words = self.word_tokenize_preprocessed(text)

        freqTable = dict()
        for word in words:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

        return freqTable


    def _score_sentences(self, sentences, freqTable) -> dict:
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
                if word in freqTable:
                    sentScore += freqTable[word]

            if len(words) > 0:
                sentScore /= len(words)
            sentenceScores.append(sentScore)
            sentIndex += 1

        return sentenceScores

    def summarize(self, text, n):
        """
        Returns summary with n sentences
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % (n, len(sentences)))
        preprText = self.preprocess_document(text)
        freqTable = self._create_frequency_table(preprText)
        # print({k: v for k, v in sorted(freqTable.items(), key=lambda item: item[1], reverse=True)})
        sentenceScores = np.array(self._score_sentences(sentences, freqTable))
        nBestIndexes = np.argpartition(sentenceScores, -n)[-n:]  # indexes of sentences with n best scores
        nBestIndexes = sorted(nBestIndexes)

        summary = ''
        for index in nBestIndexes:
            summary += sentences[index] + " "

        return summary[:-1]  # remove last space

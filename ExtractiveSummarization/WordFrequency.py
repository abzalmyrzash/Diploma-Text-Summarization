from .BaseSummarizer import BaseSummarizer
from nltk.tokenize import sent_tokenize
import numpy as np

class WordFrequency(BaseSummarizer):

    def create_frequency_table(self, text) -> dict:
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


    def score_sentences(self, sentences, freqTable) -> dict:
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
                if word in freqTable:
                    sentScore += freqTable[word]
            if len(words) > 0:
                sentScore /= len(words)
            sentenceScores.append(sentScore)

        return np.array(sentenceScores)


    def summarize(self, text, n):
        """
        Returns summary with n sentences using Word Frequency algorithm
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        self.validate_summary_length(sentences, n)
        preprText = self.preprocess_document(text)
        freqTable = self.create_frequency_table(preprText)
        sentenceScores = self.score_sentences(sentences, freqTable)

        return self.summary_from_sentence_scores(sentences, sentenceScores, n)

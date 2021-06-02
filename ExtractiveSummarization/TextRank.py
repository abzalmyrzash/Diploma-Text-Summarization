# TextRank algorithm
from .BaseSummarizer import BaseSummarizer
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import combinations
import networkx as nx
import math

class TextRank(BaseSummarizer):

    def similarity(self, s1, s2):
        """
        :param s1: words of sentence s1
        :param s2: words of sentence s2
        :return float: sentence similarity between s1 and s2
        """
        if not len(s1) or not len(s2):
            return 0.0
        return len(s1.intersection(s2))/(math.log(len(s1)+1) + math.log(len(s2)+1))


    def text_rank(self, text):
        """
        :param text: Source text
        :return list: List of sentence PageRank (TextRank) scores
        """
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [0]
        words = [set(self.word_tokenize_preprocessed(sentence)) for sentence in sentences]
        # List of words with applied stemming, without stop words and in lowercase letters.
        # An element of the list is a set of one sentence's words

        # Sentence numeration from 0 to len(sent) and generation of every possible combination by 2:
        pairs = combinations(range(len(sentences)), 2)
        similarity = [(i, j, self.similarity(words[i], words[j])) for i, j in pairs]
        similarity = filter(lambda x: x[2], similarity)  # Filter out sentences with 0 similarity

        g = nx.Graph()
        g.add_weighted_edges_from(similarity)  # Create graph with weighted edges
        scores = nx.pagerank(g)

        return [scores[i] if i in scores else 0 for i in range(len(sentences))]


    def summarize(self, text, n):
        """
        Returns summary with n sentences using TextRank algorithm
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        self.validate_summary_length(sentences, n)
        preprText = self.preprocess_document(text)
        sentenceScores = self.text_rank(preprText)
        return self.summary_from_sentence_scores(sentences, sentenceScores, n)

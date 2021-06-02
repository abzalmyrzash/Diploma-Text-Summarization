# TextRank algorithm
from itertools import combinations
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import networkx as nx
import math

class TextRank:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def similarity(self, s1, s2):
        """
        :param s1: words of sentence s1
        :param s2: words of sentence s2
        :return float: sentence similarity between s1 and s2
        """
        if not len(s1) or not len(s2):
            return 0.0
        return len(s1.intersection(s2))/(math.log(len(s1)+1) + math.log(len(s2)+1))


    def text_rank(self, sentences):
        """
        :param sentences: sentences of source text
        :return list: List of tuples (sentence index i, sentence rank pr[i], sentence s),
                      in descending order of rank
        """
        if len(sentences) < 2:
            s = sentences[0]
            return [(1, 0, s)]
        words = [set(self.stemmer.stem(word) for word in word_tokenize(sentence.lower())
                     if word not in self.stop_words) for sentence in sentences]
        # List of words with applied stemming, without stop words and in lowercase letters.
        # An element of the list is a set of one sentence's words

        # Sentence numeration from 0 to len(sent) and generation of every possible combination by 2:
        pairs = combinations(range(len(sentences)), 2)
        scores = [(i, j, self.similarity(words[i], words[j])) for i, j in pairs]
        scores = filter(lambda x: x[2], scores)  # Filter out sentences with 0 similarity

        g = nx.Graph()
        g.add_weighted_edges_from(scores)  # Create graph with weights as edges
        pr = nx.pagerank(g)  # Dict: key - vertex index, val - vertex rank

        return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr),
                      key=lambda x: pr[x[0]], reverse=True)  # Sort in descending order of rank


    def summarize(self, text, n=5):
        sentences = sent_tokenize(text)
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % (n, len(sentences)))
        tr = self.text_rank(sentences)
        top_n = sorted(tr[:n])  # Order first n sentences as in source text
        return ' '.join(x[2] for x in top_n)  # Join sentences

from nltk.tokenize import sent_tokenize
from .BaseSummarizer import BaseSummarizer
import random

class RandomExtractor(BaseSummarizer):
    def __init__(self, seed=None):
        random.seed(seed)  # if seed = None, current time is used

    def summarize(self, text, n):
        """
        Returns summary with n randomly selected sentences
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        self.validate_summary_length(sentences, n)
        nIndexes = random.sample(range(len(sentences)), n)  # n random sentence indexes
        nIndexes = sorted(nIndexes)  # sort indexes

        summary = ''
        for index in nIndexes:
            summary += sentences[index] + " "

        return summary[:-1]  # remove last space

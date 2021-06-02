import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import random

class RandomExtractor:
    def summarize(self, text, n):
        """
        Returns summary with n randomly selected sentences
        :param text: Source text
        :param n: number of sentences in summary
        """
        sentences = sent_tokenize(text)
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % (n, len(sentences)))
        nIndexes = random.sample(range(len(sentences)), n)  # n random sentence indexes
        nIndexes = sorted(nIndexes)  # sort indexes

        summary = ''
        for index in nIndexes:
            summary += sentences[index] + " "

        return summary[:-1]  # remove last space

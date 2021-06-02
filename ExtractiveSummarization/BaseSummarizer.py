# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re


class BaseSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.contraction_mapping = {
            "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
            "could've": "could have", "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not", "hadn't": "had not","hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
            "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
            "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
            "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
            "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
            "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
            "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
            "she'll've": "she will have", "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
            "so's": "so as", "this's": "this is","that'd": "that would",
            "that'd've": "that would have", "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is", "here's": "here is",
            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
            "they'll've": "they will have", "they're": "they are", "they've": "they have",
            "to've": "to have", "wasn't": "was not", "we'd": "we would","we'd've": "we would have",
            "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
            "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
            "when've": "when have", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who'll've": "who will have",
            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
            "will've": "will have", "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
            "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
            "y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have"
        }

    def preprocess_document(self, document):
        newDoc = document.replace('’', '\'')
        words = newDoc.split(" ")
        newWords = []
        for w in words:
            if w in self.contraction_mapping:
                newWords.append(self.contraction_mapping[w])
            else:
                newWords.append(w)
        newDoc = ' '.join(newWords)

        # newDoc = re.sub('[^A-Za-z .-]+', '', newDoc)
        return newDoc

    def word_tokenize_preprocessed(self, text):
        words = word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words if word.isalnum() and \
                 word not in self.stop_words]
        # words = [self.stemmer.stem(word) for word in words]
        return words

    def validate_summary_length(self, sentences, n):
        if n <= 0:
            raise ValueError("Parameter 'n' in summarize() must be greater than 0.")
        if len(sentences) < n:
            raise ValueError("Cannot extract %s sentences from text with %s sentences" % \
                             (n, len(sentences)))

    def summary_from_sentence_scores(self, sentences, sentenceScores, n):
        # indexes of sentences with n best scores
        nBestIndexes = np.argpartition(sentenceScores, -n)[-n:]
        nBestIndexes = sorted(nBestIndexes)
        summary = ' '.join((sentences[index] for index in nBestIndexes))
        return summary

    def summarize(self, text, n):
        """
        Returns summary with n sentences
        Must be implemented by child classes
        :param text: Source text
        :param n: number of sentences in summary
        """
        raise NotImplementedError("Summarize method should be implemented.")

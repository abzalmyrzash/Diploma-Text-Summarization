import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
import re
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np


class TextRankUSE:
	"""
	The difference from the other TextRank algorithm is that here Universal Sentence
	Encoder (USE) model from TensorFlow is used to calculate sentence similarity.
	Visit https://tfhub.dev/google/universal-sentence-encoder/4 to see details.
	The model weighs 1 GB so it might take quite a lot of time to download.
	"""
	def __init__(self):
		module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
		self.embed = hub.load(module_url)
		# Reduce logging output.
		# tf.logging.set_verbosity(tf.logging.ERROR)

	def summarize(self, text, n):
		sentences = sent_tokenize(text)
		message_embeddings = self.embed(sentences)


		#generate cosine similarity matrix
		sim_matrix = cosine_similarity(message_embeddings)

		#create graph and generate scores from pagerank algorithms
		nx_graph = nx.from_numpy_array(sim_matrix)
		scores = nx.pagerank(nx_graph)
		scores = list(scores.values())

		# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
		nBestIndexes = np.argpartition(scores, -n)[-n:] # indexes of sentences with n best scores
		nBestIndexes = sorted(nBestIndexes)

		summary = ''
		for index in nBestIndexes:
			summary += sentences[index] + " "

		return summary[:-1]  # remove last space
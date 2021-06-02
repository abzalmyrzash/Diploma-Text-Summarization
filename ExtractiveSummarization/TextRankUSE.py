from .BaseSummarizer import BaseSummarizer
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np


class TextRankUSE(BaseSummarizer):
	"""
	The difference from the other TextRank algorithm is that here Universal Sentence
	Encoder (USE) model from TensorFlow is used to calculate sentence similarity.
	Visit https://tfhub.dev/google/universal-sentence-encoder/4 to see details.
	The model weighs 1 GB so it might take quite a lot of time to download.
	"""
	def __init__(self):
		super().__init__()
		module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
		self.embed = hub.load(module_url)

	def text_rank(self, text):
		sentences = sent_tokenize(text)
		message_embeddings = self.embed(sentences)

		#generate cosine similarity matrix
		sim_matrix = cosine_similarity(message_embeddings)

		#create graph and generate scores from pagerank algorithms
		nx_graph = nx.from_numpy_array(sim_matrix)
		scores = nx.pagerank_numpy(nx_graph)

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
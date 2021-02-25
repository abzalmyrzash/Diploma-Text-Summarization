import math
import glob
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

SENT_ID_LENGTH = 15 # number of first characters in a sentence that will be used as an "ID"

def read_documents():
    docs = dict()
    files_list = glob.glob('texts\\*.txt')
    for filename in files_list:
        # print(filename)
        f = open(filename, 'r', encoding='utf-8')
        doc = f.read()
        # print(doc)
        f.close()
        slash_index = filename.rfind('\\')
        name = filename[slash_index+1:-4]
        # print(name)
        docs[name] = doc

    return docs

def preprocess_document(document):
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('…', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')
    return document

def preprocess_words(document):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(document)
    ps = PorterStemmer()

    nonStopWords = []
    for word in words:
        word = ps.stem(word)
        word = word.lower()
        if word not in stopWords:
            nonStopWords.append(word)

    return nonStopWords

def get_unique_words(words):
    return list(set(words))

def _create_tf_table(words) -> dict:

    freqTable = dict()
    tfTable = dict()

    totalWords = len(words)
    for word in words:
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    
    uniqueWords = get_unique_words(words)
    for word in uniqueWords:
        tfTable[word] = freqTable[word] / float(totalWords)

    return tfTable


def _create_idf_table(uniqueWords, documents):
    doc_per_word_table = dict() # in how many documents does a word occur
    for word in words:
        doc_per_word_table[word] = 1 # not 0 because 'documents' list doesn't contain our main document
    
    for doc in documents:
        doc_words = preprocess_words(documents[doc])

        for word in uniqueWords:
            if word in doc_words:
                doc_per_word_table[word] += 1

    total_documents = len(documents) + 1
    idf_table = dict()

    for word in uniqueWords:
        idf_table[word] = math.log10(total_documents / float(doc_per_word_table[word]))

    return idf_table



def _create_tf_idf_table(words, documents):
    tf_idf_table = dict()

    tf_table = _create_tf_table(words)
    uniqueWords = get_unique_words(words)
    idf_table = _create_idf_table(uniqueWords, documents)

    for word in uniqueWords:
        tf_idf_table[word] = tf_table[word] * idf_table[word]

    return tf_idf_table


def _score_sentences(sentences, tf_idf_table) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        sent_ID = sentence[:SENT_ID_LENGTH]
        sentenceValue[sent_ID] = 0
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for word in tf_idf_table:
            if word in sentence.lower():
                sentenceValue[sent_ID] += tf_idf_table[word]

        sentenceValue[sent_ID] = sentenceValue[sent_ID] / word_count_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        sent_ID = sentence[:SENT_ID_LENGTH]
        if sent_ID in sentenceValue and sentenceValue[sent_ID] > (threshold):
            summary += sentence + " "
            sentence_count += 1

    summary = summary[:-1] # remove last space
    return summary

documents = read_documents() # read documents from 'texts/' folder


# print document names and select document to summarize
docNames = list(documents.keys())
print("Enter a document's number to summarize")
for i in range(len(documents)):
    print(i+1, "-", docNames[i])

index = int(input()) - 1
docName = docNames[index]
print("You selected", docName, "for summarization")
main_document = documents.pop(docName)  # remove our selected main document from 'documents'
                                                # for optimization reason

main_document_prepr = preprocess_document(main_document)
words = preprocess_words(main_document_prepr)
# tf_table = _create_tf_table(words)
# idf_table = _create_idf_table(get_unique_words(words), documents)
tf_idf_table = _create_tf_idf_table(words, documents)
# print({k: v for k, v in sorted(tf_idf_table.items(), key=lambda item: item[1], reverse=True)})

sentences = sent_tokenize(main_document)
sentence_scores = _score_sentences(sentences, tf_idf_table)
# print(sentence_scores)
threshold = _find_average_score(sentence_scores)
summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
print(summary)

f = open('summaries/'+docName+'.txt', 'w')
f.write(summary)
f.close()
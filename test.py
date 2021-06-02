from ExtractiveSummarization import *
from nltk.tokenize import sent_tokenize
import glob


def read_documents(folderPath):
    docs = []
    docNames = []

    filesList = glob.glob(folderPath + '*.txt')
    for filename in filesList:
        f = open(filename, 'r', encoding='utf-8')
        doc = f.read()
        docs.append(doc)
        f.close()
        slash_index = filename.rfind('\\')
        name = filename[slash_index+1:-4]
        docNames.append(name)

    return (docs, docNames)

def read_document(filename):
    f = open(filename, "r", encoding='utf-8')
    text = f.read()
    f.close()
    return text

def write_summary(summary, filename):
    print(summary)
    f = open(filename, "w", encoding='utf-8')
    f.write(summary)
    f.close()


ARTICLES_FOLDER_PATH = './data/input/'
SUMMARIES_FOLDER_PATH = './data/output/'
ALGORITHMS = ['WordFrequency', 'TF-IDF', 'TextRank', 'TextRankUSE', 'LSA', 'RandomExtractor']
WF_INDEX = 0
TF_IDF_INDEX = 1
TR_INDEX = 2
TR_USE_INDEX = 3
LSA_INDEX = 4
RAND_EXT_INDEX = 5

documents, docNames = read_documents(ARTICLES_FOLDER_PATH)

# CHOOSE ALGORITHM

print("Algorithms:")
for i in range(len(ALGORITHMS)):
    print('%s. %s' % (i + 1, ALGORITHMS[i]))

algo_index = int(input("Choose algorithm: ")) - 1
print("You chose %s" % ALGORITHMS[algo_index])

if algo_index == WF_INDEX:
    summarizer = WordFrequency()
elif algo_index == TF_IDF_INDEX:
    summarizer = TF_IDF(documents)
elif algo_index == TR_INDEX:
    summarizer = TextRank()
elif algo_index == TR_USE_INDEX:
    summarizer = TextRankUSE()
elif algo_index == LSA_INDEX:
    summarizer = LSA()
elif algo_index == RAND_EXT_INDEX:
    summarizer = RandomExtractor()

# WRITE COMPRESSION RATIO
compression = float(input("Compression ratio % (if you want CR for each summary separately, write 0): "))
separate_compression = compression == 0

# SUMMARIZATION
for i in range(len(documents)):
    print(f"--- {docNames[i]} ---")
    doc = documents[i]
    num_sentences = len(sent_tokenize(doc))
    print("Total number of sentences: %s" % num_sentences)
    if separate_compression:
        compression = float(input("Compression ratio: "))
    n = round(num_sentences * compression / 100)
    print("Summary:")
    try:
        summary = summarizer.summarize(doc, n)
        write_summary(summary, f'{SUMMARIES_FOLDER_PATH}{ALGORITHMS[algo_index]}/{docNames[i]}.txt')
    except Exception as e:
        print("When summarizing \"%s\", the following exception occurred:\n%s: %s" %
              (docNames[i], type(e).__name__, e))

    print("\n\n")
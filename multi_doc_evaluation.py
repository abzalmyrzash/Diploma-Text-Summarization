print("Importing...")
from ExtractiveSummarization.WordFrequency import WordFrequency
from ExtractiveSummarization.TF_IDF import TF_IDF
from ExtractiveSummarization.TextRank import TextRank
from ExtractiveSummarization.LSA import LSA
from numpy import array
from nltk.tokenize import sent_tokenize
from ROUGE.ROUGE import rouge_n, rouge_l, rouge_s
from ROUGE.ROUGE_stem import rouge_n_s, rouge_l_s, rouge_s_s
from ROUGE.ROUGE_stem_sw import rouge_n_sw, rouge_l_sw, rouge_s_sw
import dataset
import glob
import time
import os

def print2cf(string, file):  # print to both console and file
    print(string)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(string)

CNN_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/stories/'
CNN_QUARANTINE_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/quarantine/'
CNN_RESULTS_FOLDER_PATH = './data/ROUGE results (cnn)/'
DM_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/dailymail/stories/'
DM_QUARANTINE_FOLDER_PATH = './../datasets/CNN_DailyMail/dailymail/quarantine/'
DM_RESULTS_FOLDER_PATH = './data/ROUGE results (dm)/'
ALGORITHMS = ['WordFrequency', 'TF-IDF', 'TextRank', 'LSA']
WF_INDEX = 0
TF_IDF_INDEX = 1
TR_INDEX = 2
LSA_INDEX = 3

COLLECTION_SIZE = 1000

selected = False
while not selected:
    print("Which stories folder to test summarizer on?")
    print("1. CNN stories")
    print("2. Daily Mail stories")
    try:
        stories_choice = int(input())
    except:
        print("Please enter an integer!")
        continue

    selected = True
    if stories_choice == 1:
        STORIES_FOLDER_PATH = CNN_STORIES_FOLDER_PATH
        RESULTS_FOLDER_PATH = CNN_RESULTS_FOLDER_PATH
        QUARANTINE_FOLDER_PATH = CNN_QUARANTINE_FOLDER_PATH
    elif stories_choice == 2:
        STORIES_FOLDER_PATH = DM_STORIES_FOLDER_PATH
        RESULTS_FOLDER_PATH = DM_RESULTS_FOLDER_PATH
        QUARANTINE_FOLDER_PATH = DM_QUARANTINE_FOLDER_PATH
    else:
        print("Invalid stories folder number!")
        selected = False


print("Loading list of story file names...")
story_files = glob.glob(STORIES_FOLDER_PATH + '*.story')
num_stories = len(story_files)
print("%s story file names loaded." % num_stories)

# CHOOSE ALGORITHM
algo_index = TF_IDF_INDEX

if algo_index == WF_INDEX:
    summarizer = WordFrequency()
elif algo_index == TF_IDF_INDEX:
    preBuildIdf = input("Pre-build IDF table? (type 'YES' for yes and anything else for no)")
    preBuildIdf = True if preBuildIdf == 'YES' else False
    print(preBuildIdf)
elif algo_index == TR_INDEX:
    summarizer = TextRank()
elif algo_index == LSA_INDEX:
    summarizer = LSA()
else:
    print("Invalid algorithm number!")
    selected = False


results_path = RESULTS_FOLDER_PATH + ALGORITHMS[algo_index] + '.txt'

start_time = time.time()
sum_rouge_1 = array([0.0, 0.0, 0.0])
sum_rouge_2 = array([0.0, 0.0, 0.0])
sum_rouge_l = array([0.0, 0.0, 0.0])
sum_rouge_s = array([0.0, 0.0, 0.0])
sum_rouge_s4 = array([0.0, 0.0, 0.0])
sum_rouge_s9 = array([0.0, 0.0, 0.0])

sum_rouge_1_s = array([0.0, 0.0, 0.0])
sum_rouge_2_s = array([0.0, 0.0, 0.0])
sum_rouge_l_s = array([0.0, 0.0, 0.0])
sum_rouge_s_s = array([0.0, 0.0, 0.0])
sum_rouge_s4_s = array([0.0, 0.0, 0.0])
sum_rouge_s9_s = array([0.0, 0.0, 0.0])

sum_rouge_1_sw = array([0.0, 0.0, 0.0])
sum_rouge_2_sw = array([0.0, 0.0, 0.0])
sum_rouge_l_sw = array([0.0, 0.0, 0.0])
sum_rouge_s_sw = array([0.0, 0.0, 0.0])
sum_rouge_s4_sw = array([0.0, 0.0, 0.0])
sum_rouge_s9_sw = array([0.0, 0.0, 0.0])

count_files = 0
for story_file in story_files:
    if count_files % COLLECTION_SIZE == 0:
        documents = []
        count_documents = 0
        print("Loading documents...")
        for i in range(count_files, count_files + COLLECTION_SIZE):
            doc_file = story_files[i]
            article, abstract = dataset.get_art_abs(doc_file)
            documents.append(article)
            print("Loaded %s out of %s documents..." % (i + 1, COLLECTION_SIZE), \
                   end='\r', flush=True)
        print(flush=True)
        print("Initializing summarizer...")
        summarizer = TF_IDF(documents, preBuildIdf)

    article, abstract = dataset.get_art_abs(story_file)

    # if article == '':  # some files don't have article text
    #     print("%s does not have article text" % story_file)
    #     continue
    abstract_sent = sent_tokenize(abstract)
    n = len(abstract_sent)
    
    try:
        eval_text = summarizer.summarize(article, count_files, n)
    except ValueError as e:
        sf_basename = os.path.basename(story_file)
        print("%s: %s" % (sf_basename, e))
        os.rename(story_file, QUARANTINE_FOLDER_PATH + sf_basename)
        print("%s moved to %s" % (sf_basename, QUARANTINE_FOLDER_PATH))
        continue
    except KeyError as e:
        sf_basename = os.path.basename(story_file)
        print("%s: %s" % (sf_basename, e))
    finally:
        count_files += 1

    sum_rouge_1 += rouge_n(eval_text, abstract, 1)
    sum_rouge_2 += rouge_n(eval_text, abstract, 2)
    sum_rouge_l += rouge_l(eval_text, abstract)
    sum_rouge_s += rouge_s(eval_text, abstract, 10000)
    sum_rouge_s4 += rouge_s(eval_text, abstract, 4)
    sum_rouge_s9 += rouge_s(eval_text, abstract, 9)

    sum_rouge_1_s += rouge_n_s(eval_text, abstract, 1)
    sum_rouge_2_s += rouge_n_s(eval_text, abstract, 2)
    sum_rouge_l_s += rouge_l_s(eval_text, abstract)
    sum_rouge_s_s += rouge_s_s(eval_text, abstract, 10000)
    sum_rouge_s4_s += rouge_s_s(eval_text, abstract, 4)
    sum_rouge_s9_s += rouge_s_s(eval_text, abstract, 9)

    sum_rouge_1_sw += rouge_n_sw(eval_text, abstract, 1)
    sum_rouge_2_sw += rouge_n_sw(eval_text, abstract, 2)
    sum_rouge_l_sw += rouge_l_sw(eval_text, abstract)
    sum_rouge_s_sw += rouge_s_sw(eval_text, abstract, 10000)
    sum_rouge_s4_sw += rouge_s_sw(eval_text, abstract, 4)
    sum_rouge_s9_sw += rouge_s_sw(eval_text, abstract, 9)

    print("%s out of %s stories summarized" % (count_files, num_stories), end='\r', flush=True)

    if count_files % 1000 == 0:
        output = ''

        output += "ROUGE-1     %s" % (sum_rouge_1/count_files) + '\n'
        output += "ROUGE-2     %s" % (sum_rouge_2/count_files) + '\n'
        output += "ROUGE-L     %s" % (sum_rouge_l/count_files) + '\n'
        output += "ROUGE-S     %s" % (sum_rouge_s/count_files) + '\n'
        output += "ROUGE-S4    %s" % (sum_rouge_s4/count_files) + '\n'
        output += "ROUGE-S9    %s" % (sum_rouge_s9/count_files) + '\n'

        output += "ROUGE-1-s   %s" % (sum_rouge_1_s/count_files) + '\n'
        output += "ROUGE-2-s   %s" % (sum_rouge_2_s/count_files) + '\n'
        output += "ROUGE-L-s   %s" % (sum_rouge_l_s/count_files) + '\n'
        output += "ROUGE-S-s   %s" % (sum_rouge_s_s/count_files) + '\n'
        output += "ROUGE-S4-s  %s" % (sum_rouge_s4_s/count_files) + '\n'
        output += "ROUGE-S9-s  %s" % (sum_rouge_s9_s/count_files) + '\n'

        output += "ROUGE-1-sw  %s" % (sum_rouge_1_sw/count_files) + '\n'
        output += "ROUGE-2-sw  %s" % (sum_rouge_2_sw/count_files) + '\n'
        output += "ROUGE-L-sw  %s" % (sum_rouge_l_sw/count_files) + '\n'
        output += "ROUGE-S-sw  %s" % (sum_rouge_s_sw/count_files) + '\n'
        output += "ROUGE-S4-sw %s" % (sum_rouge_s4_sw/count_files) + '\n'
        output += "ROUGE-S9-sw %s" % (sum_rouge_s9_sw/count_files) + '\n'

        output += "--- %s seconds ---" % (time.time() - start_time)

        print2cf(output, results_path)

output = ''

output += "ROUGE-1     %s" % (sum_rouge_1/count_files) + '\n'
output += "ROUGE-2     %s" % (sum_rouge_2/count_files) + '\n'
output += "ROUGE-L     %s" % (sum_rouge_l/count_files) + '\n'
output += "ROUGE-S     %s" % (sum_rouge_s/count_files) + '\n'
output += "ROUGE-S4    %s" % (sum_rouge_s4/count_files) + '\n'
output += "ROUGE-S9    %s" % (sum_rouge_s9/count_files) + '\n'

output += "ROUGE-1-s   %s" % (sum_rouge_1_s/count_files) + '\n'
output += "ROUGE-2-s   %s" % (sum_rouge_2_s/count_files) + '\n'
output += "ROUGE-L-s   %s" % (sum_rouge_l_s/count_files) + '\n'
output += "ROUGE-S-s   %s" % (sum_rouge_s_s/count_files) + '\n'
output += "ROUGE-S4-s  %s" % (sum_rouge_s4_s/count_files) + '\n'
output += "ROUGE-S9-s  %s" % (sum_rouge_s9_s/count_files) + '\n'

output += "ROUGE-1-sw  %s" % (sum_rouge_1_sw/count_files) + '\n'
output += "ROUGE-2-sw  %s" % (sum_rouge_2_sw/count_files) + '\n'
output += "ROUGE-L-sw  %s" % (sum_rouge_l_sw/count_files) + '\n'
output += "ROUGE-S-sw  %s" % (sum_rouge_s_sw/count_files) + '\n'
output += "ROUGE-S4-sw %s" % (sum_rouge_s4_sw/count_files) + '\n'
output += "ROUGE-S9-sw %s" % (sum_rouge_s9_sw/count_files) + '\n'

output += "--- %s seconds ---" % (time.time() - start_time)

print2cf(output, results_path)

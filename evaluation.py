print("Importing...")
from ExtractiveSummarization import WordFrequency
from ExtractiveSummarization import TF_IDF
from ExtractiveSummarization import TextRank
from ExtractiveSummarization import TextRankUSE
from ExtractiveSummarization import LSA
from ExtractiveSummarization import RandomExtractor
import ExtractiveSummarization
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

print("WARNING: In case you haven't done it already, please run quarantine.py"
      "and build_save_tf_idf_summarizer.py before running this program.\n"
      "Running quarantine.py will move story files that contain articles shorter"
      "than their summaries to folders 'cnn/quarantine' and 'dailymail/quarantine'.\n"
      "Running build_save_tf_idf_summarizer.py will create a TF_IDF summarizer with"
      "pre-calculated doc_per_word_table using all of the story files.")

proceed = input("Proceed? (type 'YES' for yes, anything else for no)")
if proceed != 'YES':
    return

CNN_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/stories/'
CNN_SUMMARIES_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/summaries/'
CNN_RESULTS_FOLDER_PATH = './data/ROUGE results (cnn)/'

DM_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/dailymail/stories/'
DM_SUMMARIES_FOLDER_PATH = './../datasets/CNN_DailyMail/daiymail/summaries/'
DM_RESULTS_FOLDER_PATH = './data/ROUGE results (dm)/'

TF_IDF_DATA_PATH = './data/TF_IDF_data.bin'

ALGORITHMS = ['WordFrequency', 'TF-IDF', 'TextRank', 'TextRankUSE', 'LSA', 'RandomExtractor']
WF_INDEX = 0
TF_IDF_INDEX = 1
TR_INDEX = 2
TR_USE_INDEX = 3
LSA_INDEX = 4
RAND_EXT_INDEX = 5

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
        SUMMARIES_FOLDER_PATH = CNN_SUMMARIES_FOLDER_PATH
    elif stories_choice == 2:
        STORIES_FOLDER_PATH = DM_STORIES_FOLDER_PATH
        RESULTS_FOLDER_PATH = DM_RESULTS_FOLDER_PATH
        QUARANTINE_FOLDER_PATH = DM_QUARANTINE_FOLDER_PATH
        SUMMARIES_FOLDER_PATH = DM_SUMMARIES_FOLDER_PATH
    else:
        print("Invalid stories folder number!")
        selected = False


print("Loading list of story file names...")
story_files = glob.glob(STORIES_FOLDER_PATH + '*.story')
num_stories = len(story_files)
print("%s story file names loaded." % num_stories)

# CHOOSE ALGORITHM

selected = False
while not selected:
    print("Summarization algorithms:")
    for i in range(len(ALGORITHMS)):
        print('%s. %s' % (i + 1, ALGORITHMS[i]))

    try:
        algo_index = int(input("Choose algorithm: ")) - 1
    except:
        print("Please enter an integer!")
        continue

    print("You chose %s" % ALGORITHMS[algo_index])
    selected = True

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
    else:
        print("Invalid algorithm number!")
        selected = False



writeSummary = input("Write generated summaries to files? (type 'YES' for yes, anything else for no)")
writeSummary = True if writeSummary == 'YES' else False

results_path = RESULTS_FOLDER_PATH + ALGORITHMS[algo_index] + '.txt'

# n = 10
# for i in range(n):
#     story_file = story_files[i]
#     article, abstract = dataset.get_art_abs(story_file)
#     print(f"{i + 1} out of {n}\n")
    # print("\n-----ARTICLE-----\n")
    # print(article)
    # print("\n-----ABSTRACT-----\n")
    # print(abstract)
    # print("\n=====================================================================================\n")


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
    sf_basename = os.path.basename(story_file)
    article, abstract = dataset.get_art_abs(story_file)
    abstract_sent = sent_tokenize(abstract)
    n = len(abstract_sent)
    
    try:
        eval_text = summarizer.summarize(article, n)
    except ValueError as e:
        print(f"When summarizing {story_file}, the following exception occurred:\n{e}")
        continue
    finally:
        count_files += 1

    summary_file = SUMMARIES_FOLDER_PATH + ALGORITHMS[algo_index] + '/' + sf_basename
    if writeSummary:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("--- ORIGINAL SUMMARY ---\n")
            f.write(abstract)
            f.write("\n--- GENERATED SUMMARY ---\n")
            f.write(eval_text)

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
        print(flush=True)
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

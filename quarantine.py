print("Importing...")
from numpy import array
from nltk.tokenize import sent_tokenize
import dataset
import glob
import time
import os

CNN_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/stories/'
CNN_QUARANTINE_FOLDER_PATH = './../datasets/CNN_DailyMail/cnn/quarantine/'
DM_STORIES_FOLDER_PATH = './../datasets/CNN_DailyMail/dailymail/stories/'
DM_QUARANTINE_FOLDER_PATH = './../datasets/CNN_DailyMail/dailymail/quarantine/'
STORIES_FOLDER_PATHS = [CNN_STORIES_FOLDER_PATH, DM_STORIES_FOLDER_PATH]
QUARANTINE_FOLDER_PATHS = [CNN_QUARANTINE_FOLDER_PATH, DM_QUARANTINE_FOLDER_PATH]

for i in range(2):
	STORIES_FOLDER_PATH = STORIES_FOLDER_PATHS[i]
	QUARANTINE_FOLDER_PATH = QUARANTINE_FOLDER_PATHS[i]

	if not os.path.exists(STORIES_FOLDER_PATH):
		os.makedirs(STORIES_FOLDER_PATH)
		print(f"Directory {STORIES_FOLDER_PATH} created.")

	print("Loading list of story file names...")
	story_files = glob.glob(STORIES_FOLDER_PATH + '*.story')
	already_quarantined = glob.glob(QUARANTINE_FOLDER_PATH + '*.story')
	num_stories = len(story_files)
	print("%s story file names loaded." % num_stories)
	print("%s stories are already quarantined." % len(already_quarantined))


	start_time = time.time()

	count_files = 0

	for story_file in story_files:
		article, abstract = dataset.get_art_abs(story_file)

		# if article == '':  # some files don't have article text
		#     print("%s does not have article text" % story_file)
		#     continue
		article_sent = sent_tokenize(article)
		abstract_sent = sent_tokenize(abstract)
		art_len = len(article_sent)
		abs_len = len(abstract_sent)

		if art_len < abs_len:
			sf_basename = os.path.basename(story_file)
			print("%s - art: %s, abs: %s" % (sf_basename, art_len, abs_len))
			os.rename(story_file, QUARANTINE_FOLDER_PATH + sf_basename)
			print("%s moved to %s" % (sf_basename, QUARANTINE_FOLDER_PATH))

		count_files += 1

		print("%s out of %s story files have been checked..." % (count_files, num_stories), end='\r', flush=True)

		if count_files % 1000 == 0:
			print(count_files)
			print("--- %s seconds ---\n" % (time.time() - start_time))

	print("\n==============================================================================\n")
	print("%s story files have been quarantined." % (len(story_files) - count_files))
	print("--- %s seconds ---" % (time.time() - start_time))
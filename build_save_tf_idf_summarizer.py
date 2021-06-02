from ExtractiveSummarization import TF_IDF
import glob
import dataset
import pickle
import os

CNN_STORIES_PATH = './../datasets/CNN_DailyMail/cnn/stories/'
DM_STORIES_PATH = './../datasets/CNN_DailyMail/dailymail/stories/'
TF_IDF_DATA_PATH = './data/TF_IDF_data.bin'

print("Loading file names...")
cnn_story_files = glob.glob(CNN_STORIES_PATH + '*.story')
dm_story_files = glob.glob(DM_STORIES_PATH + '*.story')
story_files = cnn_story_files + dm_story_files

# if tfIdfSummarizer was already saved in the file, load and continue from there
# else build a new one and start from scratch
if os.path.exists(TF_IDF_DATA_PATH):
	print(f"Loading TF-IDF summarizer from {TF_IDF_DATA_PATH}")
	with open(TF_IDF_DATA_PATH, 'rb') as f:
		tfIdfSummarizer = pickle.load(f)
		alreadyPrDocs = tfIdfSummarizer.total_documents
	print(f"The loaded TF-IDF summarizer has already processed {alreadyPrDocs} stories.")
	print(f"Start processing from the {alreadyPrDocs + 1}th story.")
else:
	print("Building TF-IDF summarizer...")
	tfIdfSummarizer = TF_IDF()
	alreadyPrDocs = 0

total_stories = len(story_files)
for i in range(alreadyPrDocs, total_stories):
	story_file = story_files[i]
	article, abstract = dataset.get_art_abs(story_file)
	tfIdfSummarizer.update_dpw_table([article])
	print(f"{i + 1} out of {total_stories} stories processed...", end='\r', flush=True)
	# save every 1000 stories
	if (i + 1) % 1000 == 0:
		print(flush=True)
		print("Saving TF-IDF summarizer")
		with open(TF_IDF_DATA_PATH, 'wb') as f:
			pickle.dump(tfIdfSummarizer, f)

print(flush=True)
print("Saving TF-IDF summarizer")
with open(TF_IDF_DATA_PATH, 'wb') as f:
	pickle.dump(tfIdfSummarizer, f)
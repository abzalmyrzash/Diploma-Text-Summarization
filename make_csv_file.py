import csv
import dataset
import glob

CNN_STORIES_PATH = "./../datasets/CNN_DailyMail/cnn/stories/"
DM_STORIES_PATH = "./../datasets/CNN_DailyMail/dm/stories/"
CSV_FILE_PATH = "./../datasets/CNN_DailyMail/cnn_dm.csv"

cnn_story_files = glob.glob(CNN_STORIES_PATH + "*.txt")
dm_story_files = glob.glob(DM_STORIES_PATH + "*.txt")

with open(CSV_FILE_PATH, 'w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for story_file in cnn_story_files:
		article, abstract = dataset.get_art_abs(story_file)
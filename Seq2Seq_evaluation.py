from ROUGE.ROUGE import rouge_n, rouge_l, rouge_s
# import pandas as pd
import csv
from numpy import array
import time

def print2cf(string, file):  # print to both console and file
	print(string)
	with open(file, 'w', encoding='utf-8') as f:
		f.write(string)

DATA_PATH = "./../datasets/Amazon reviews/output/Seq2Seq summaries.csv"
RESULTS_PATH = './data/Seq2Seq ROUGE results/Seq2Seq.txt'

print("Reading csv file...")

rows = []
with open(DATA_PATH) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	first_row = True
	for row in csv_reader:
		if first_row:
			print(f'Column names are {", ".join(row)}')
			first_row = False
		else:
			# row[0] is source text, we only need original and predicted summaries
			rows.append((row[1], row[2]))


print("Calculating ROUGE...")
start_time = time.time()
sum_rouge_1 = array([0.0, 0.0, 0.0])
sum_rouge_2 = array([0.0, 0.0, 0.0])
sum_rouge_l = array([0.0, 0.0, 0.0])
sum_rouge_s = array([0.0, 0.0, 0.0])
sum_rouge_s4 = array([0.0, 0.0, 0.0])
sum_rouge_s9 = array([0.0, 0.0, 0.0])

count_rows = 0
total_rows = len(rows)

for row in rows:
	abstract = row[0]
	eval_text = row[1]

	sum_rouge_1 += rouge_n(eval_text, abstract, 1)
	sum_rouge_2 += rouge_n(eval_text, abstract, 2)
	sum_rouge_l += rouge_l(eval_text, abstract)
	sum_rouge_s += rouge_s(eval_text, abstract, 10000)
	sum_rouge_s4 += rouge_s(eval_text, abstract, 4)
	sum_rouge_s9 += rouge_s(eval_text, abstract, 9)

	count_rows += 1
	print(f"{count_rows} out of {total_rows} rows evaluated.", end='\r', flush=True)
	if count_rows % 1000 == 0:
		output = ''

		output += "ROUGE-1     %s" % (sum_rouge_1/count_rows) + '\n'
		output += "ROUGE-2     %s" % (sum_rouge_2/count_rows) + '\n'
		output += "ROUGE-L     %s" % (sum_rouge_l/count_rows) + '\n'
		output += "ROUGE-S     %s" % (sum_rouge_s/count_rows) + '\n'
		output += "ROUGE-S4    %s" % (sum_rouge_s4/count_rows) + '\n'
		output += "ROUGE-S9    %s" % (sum_rouge_s9/count_rows) + '\n'

		output += "--- %s seconds ---" % (time.time() - start_time)
		print(flush=True)
		print2cf(output, RESULTS_PATH)


output = ''

output += "ROUGE-1     %s" % (sum_rouge_1/count_rows) + '\n'
output += "ROUGE-2     %s" % (sum_rouge_2/count_rows) + '\n'
output += "ROUGE-L     %s" % (sum_rouge_l/count_rows) + '\n'
output += "ROUGE-S     %s" % (sum_rouge_s/count_rows) + '\n'
output += "ROUGE-S4    %s" % (sum_rouge_s4/count_rows) + '\n'
output += "ROUGE-S9    %s" % (sum_rouge_s9/count_rows) + '\n'

output += "--- %s seconds ---" % (time.time() - start_time)
print2cf(output, RESULTS_PATH)
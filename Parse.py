import os, re
import codecs, itertools
import numpy as np
import util

#INPUT_DATA = 'scraped/data.txt'
INPUT_DATA = 'scraped/all_comments.txt'
SAVE_DIR = 'parsed_all/'

#Create directory to hold parsed data
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

def split_to_words(full_str):
	#Split clean sentence into words
	words = full_str.split(' ')

	#Remove annoying space-separated letters
	num_single_letter = 0
	for word in words:
		if len(word) == 1 and word in 'abcdefghijklmnopqrstuvwxyzx':
			num_single_letter += 1
		#If sentence contains an unusually long word, ignore it
		if len(word) >= 24:
			return []
	if num_single_letter > 5:
		return []

	return [word for word in words if len(word) > 0]

def parse_line(line):
	vid, title, comment = line[:-1].split('~')
	title = util.clean_text(title)
	comment = util.clean_text(comment)
	title_ix = split_to_words(title)
	if len(title_ix) == 0:
		return [], []
	comment_ix = split_to_words(comment)
	return title_ix, comment_ix

def words_to_ixs(words, all_words, word_to_ix):
	for word in words:
		if word not in word_to_ix:
			word_to_ix[word] = len(all_words)
			all_words.append(word)
	return [word_to_ix[w] for w in words]

#Read the file line by line
print("Parsing...")
all_title_words = []
all_comment_words = []
with codecs.open(INPUT_DATA, 'r', encoding='utf-8') as fin:
	for line in fin:
		title_words, comment_words = parse_line(line)
		if len(title_words) == 0 or len(comment_words) == 0:
			continue
		all_title_words.append(title_words)
		all_comment_words.append(comment_words)

#Generate a word frequency to help eliminate uncommon samples
print("Counting Occurrence...")
comment_word_count = {}
for comment_words in all_comment_words:
	for word in comment_words:
		if word in comment_word_count:
			comment_word_count[word] += 1
		else:
			comment_word_count[word] = 1

#Eliminate any words that appeared only once
print("Eliminating Ultra-Rare Words...")
all_title_ixs = []
all_comment_ixs = []
title_word_list = ['']
title_word_map= {'':0}
comment_word_list = ['']
comment_word_map = {'':0}
for title_words, comment_words in zip(all_title_words, all_comment_words):
	for word in comment_words:
		if comment_word_count[word] <= 1:
			break
	else:
		all_title_ixs.append(words_to_ixs(title_words, title_word_list, title_word_map))
		all_comment_ixs.append(words_to_ixs(comment_words, comment_word_list, comment_word_map))

#Generate lengths for the flattened data
print("Converting To Indices...")
all_title_lens = []
all_comment_lens = []
for title_ix in all_title_ixs:
	all_title_lens.append(len(title_ix))
for comment_ix in all_comment_ixs:
	all_comment_lens.append(len(comment_ix))
all_title_ixs = list(itertools.chain.from_iterable(all_title_ixs))
all_comment_ixs = list(itertools.chain.from_iterable(all_comment_ixs))

#Write results with numpy
print("Total Pairs: " + str(len(all_title_lens)))
print("Total Title Words: " + str(len(all_title_ixs)))
print("Total Comment Words: " + str(len(all_comment_ixs)))
np.save(SAVE_DIR + 'titles.npy', np.array(all_title_ixs, dtype=np.int32))
np.save(SAVE_DIR + 'comments.npy', np.array(all_comment_ixs, dtype=np.int32))
np.save(SAVE_DIR + 'title_lens.npy', np.array(all_title_lens, dtype=np.int32))
np.save(SAVE_DIR + 'comment_lens.npy', np.array(all_comment_lens, dtype=np.int32))

#Save dictionary of all used words
print("Title Dict Size: " + str(len(title_word_list)))
with open(SAVE_DIR + 'title_dict.txt', 'w') as fout:
	for w in title_word_list:
		fout.write(w + '\n')
print("Comment Dict Size: " + str(len(comment_word_list)))
with open(SAVE_DIR + 'comment_dict.txt', 'w') as fout:
	for w in comment_word_list:
		fout.write(w + '\n')

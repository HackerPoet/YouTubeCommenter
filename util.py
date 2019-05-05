import os, re
import numpy as np

def clean_text(text):
	for c in [u"\u0060", u"\u00B4", u"\u2018", u"\u2019"]:
		text = text.replace(c, "'")
	for c in [u"\u00C0", u"\u00C1", u"\u00C2", u"\u00C3", u"\u00C4", u"\u00C5",
	          u"\u00E0", u"\u00E1", u"\u00E2", u"\u00E3", u"\u00E4", u"\u00E5"]:
		text = text.replace(c, "a")
	for c in [u"\u00C8", u"\u00C9", u"\u00CA", u"\u00CB",
	          u"\u00E8", u"\u00E9", u"\u00EA", u"\u00EB"]:
		text = text.replace(c, "e")
	for c in [u"\u00CC", u"\u00CD", u"\u00CE", u"\u00CF",
	          u"\u00EC", u"\u00ED", u"\u00EE", u"\u00EF"]:
		text = text.replace(c, "i")
	for c in [u"\u00D2", u"\u00D3", u"\u00D4", u"\u00D5", u"\u00D6",
	          u"\u00F2", u"\u00F3", u"\u00F4", u"\u00F5", u"\u00F6"]:
		text = text.replace(c, "o")
	for c in [u"\u00DA", u"\u00DB", u"\u00DC", u"\u00DD",
	          u"\u00FA", u"\u00FB", u"\u00FC", u"\u00FD"]:
		text = text.replace(c, "u")
	text = text.replace(u"\u00D1", "n").replace(u"\u00F1", "n")
	text = text.encode('utf-8').decode()
	if 'http' in text:
		return ''
	text = re.sub(r'[^0-9a-z .,?!\'/:;<>#\-\$%&]', ' ', text.lower())
	text = ' ' + text + ' '
	text = text.replace('&', ' and ')
	text = re.sub(r'\.( +\.)+', '..', text)
	text = re.sub(r'\.\.+', ' ^ ', text)
	text = re.sub(r',+', ',', text)
	text = re.sub(r'\-+', '-', text)
	text = re.sub(r'\?+', ' ? ', text)
	text = re.sub(r'\!+', ' ! ', text)
	text = re.sub(r'\'+', "'", text)
	text = re.sub(r';+', ':', text)
	text = re.sub(r'/+', ' / ', text)
	text = re.sub(r'<+', ' < ', text)
	text = re.sub(r'>+', ' > ', text)
	text = text.replace('%', '% ')
	text = text.replace(' - ', ' : ')
	text = text.replace(' -', " - ")
	text = text.replace('- ', " - ")
	text = text.replace(" '", " ")
	text = text.replace("' ", " ")
	for c in ".,:":
		text = text.replace(c + ' ', ' ' + c + ' ')
	#text = re.sub(r' \d\d?:\d\d ', ' 0:00 ', text)
	text = re.sub(r' +', ' ', text.strip(' '))
	text = text.replace('^', '...')
	return text

def load_dict(fname, word_list, word_dict):
	with open(fname, 'r') as fin:
		for line in fin:
			line = line[:-1]
			assert(line not in word_dict)
			word_dict[line] = len(word_list)
			word_list.append(line)
	assert(word_list[0] == '')

def load_title_dict(PARSED_DIR):
	title_words = []
	title_word_to_ix = {}
	load_dict(PARSED_DIR + 'title_dict.txt', title_words, title_word_to_ix)
	print("Loaded " + str(len(title_words)) + " title word dictionary.")
	return title_words, title_word_to_ix

def load_comment_dict(PARSED_DIR):
	comment_words = []
	comment_word_to_ix = {}
	load_dict(PARSED_DIR + 'comment_dict.txt', comment_words, comment_word_to_ix)
	print("Loaded " + str(len(comment_words)) + " comment word dictionary.")
	return comment_words, comment_word_to_ix

def load_title_sentences(PARSED_DIR):
	#Load the raw data
	print("Loading Titles...")
	titles = np.load(PARSED_DIR + 'titles.npy')
	title_lens = np.load(PARSED_DIR + 'title_lens.npy')
	print("Loaded " + str(len(title_lens)) + " titles.")

	#Extract all title sentences
	title_ix = 0
	title_sentences = []
	for title_len in title_lens:
		title_sentences.append(titles[title_ix:title_ix + title_len])
		title_ix += title_len
	return title_sentences

def load_comment_sentences(PARSED_DIR):
	#Load the raw data
	print("Loading Comments...")
	comments = np.load(PARSED_DIR + 'comments.npy')
	comment_lens = np.load(PARSED_DIR + 'comment_lens.npy')
	print("Loaded " + str(len(comment_lens)) + " comments.")

	#Extract all comment sentences
	comment_ix = 0
	comment_sentences = []
	for comment_len in comment_lens:
		comment_sentences.append(comments[comment_ix:comment_ix + comment_len])
		comment_ix += comment_len
	return comment_sentences

def bag_of_words(title_ixs, title_dict_size):
	title_sample = np.zeros((title_dict_size,), dtype=np.uint8)
	for ix in title_ixs:
		title_sample[ix] = 1
	return title_sample

def create_training_samples(PARSED_DIR, seq_size, out_seq=False):
	#Load the raw data
	print("Loading Titles...")
	titles = np.load(PARSED_DIR + 'titles.npy')
	title_lens = np.load(PARSED_DIR + 'title_lens.npy')
	print("Loaded " + str(len(title_lens)) + " titles.")

	#Load the raw data
	print("Loading Comments...")
	comments = np.load(PARSED_DIR + 'comments.npy')
	comment_lens = np.load(PARSED_DIR + 'comment_lens.npy')
	print("Loaded " + str(len(comment_lens)) + " comments.")

	#Convert to training samples
	print("Creating Training Samples...")
	title_ix = 0
	comment_ix = 0
	title_ix_samples = []
	title_unique_samples = []
	past_samples = []
	pred_samples = []
	title_dict_size = np.amax(titles) + 1
	for i in range(title_lens.shape[0]):
		title_len = title_lens[i]
		title_sample = np.zeros((title_dict_size,), dtype=np.uint8)
		for j in range(title_len):
			word = titles[title_ix]
			title_sample[word] = 1
			title_ix += 1
		title_unique_samples.append(title_sample)

		comment_len = comment_lens[i]
		past_sample = np.zeros((seq_size,), dtype=np.int32)
		end_j = comment_len + 1
		if out_seq:
			end_j = max(end_j, seq_size)
		for j in range(end_j):
			if not out_seq or j >= seq_size - 1:
				title_ix_samples.append(len(title_unique_samples) - 1)
				past_samples.append(past_sample)

			if j >= comment_len:
				next_word = 0
			else:
				next_word = comments[comment_ix]
				comment_ix += 1

			past_sample = np.roll(past_sample, -1)
			past_sample[-1] = next_word
			if not out_seq:
				pred_samples.append(next_word)
			elif j >= seq_size - 1:
				pred_samples.append(past_sample)

	num_samples = len(past_samples)
	assert(title_ix == len(titles))
	assert(comment_ix == len(comments))
	assert(num_samples == len(pred_samples))
	assert(num_samples == len(title_ix_samples))
	title_ix_samples = np.array(title_ix_samples, dtype=np.int32)
	title_unique_samples = np.array(title_unique_samples, dtype=np.uint8)
	past_samples = np.array(past_samples, dtype=np.int32)
	pred_samples = np.array(pred_samples, dtype=np.int32)
	pred_samples = np.expand_dims(pred_samples, axis=-1)
	print("Created " + str(num_samples) + " samples.")

	return title_ix_samples, title_unique_samples, past_samples, pred_samples

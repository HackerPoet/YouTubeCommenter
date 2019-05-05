from urllib.request import urlopen
import os, re
import codecs
import json

#Update this line with your API key
API_KEY = ''

JUSTIN_Y_CHANNEL = 'UCiTfB-A55Vq2fB610vaWJVA'
SAVE_DIR = 'scraped/'
DATA_FILE = SAVE_DIR + 'all_comments.txt'
NUM_COMMENT_PAGES = 1
READ_CACHE_ONLY = False

ALL_COMMENTS = True
ALL_COMMENTS_LEN_MIN = 16
ALL_COMMENTS_LEN_MAX = 120

ALL_PLAYLISTS = [
	'LLiTfB-A55Vq2fB610vaWJVA',           #5000 - Justin Y's liked videos
	'PLEDLY9RcAVEAIhk1PO_3HNKZnqo8Tk6b5', #195 - The Great ones
	'PLAZQuoSz85i6ZXYGUGoTny2Foj-GOAwde', #445 - the justin y. archive
	'PLDqtsMAe_KP1FOYhp2PB6Og4KkMiNaHtf', #365 - Videos that Justin Y. has commented on
	'PLdwJKVt5J4kPa4hpJb9T-ai3gCgn_aBns', #226 - Videos visited by our god king Justin Y
	'PLdCWc2HiLQ-KiJCvVAPbItZng7L3zQeoD', #116 - Justin Y. comments on here
	'PLs9Y9RDj_m8GFrXCJVvmjXXrHb3TvtEWZ', #197 - Justin Y. Comments
	'PLXzFN31Zk_rJv9pbpLoBC4tom2upYDSpC', #110 - Justin y. comments
	'PLjlDY1ZxBlPKuaikG2_PfzYhOj04WXihN', #268 - Videos that have Justin Y.
	'PL2P9kapzgxPKt1R4a1uhaSTrBBWgYliP3', #307 - Vids With Justin Y.'s Comment
	'PLflokqibVlnOJ_oJo99UaOpiTT1iX0jUc', #118 - Where Justin Y. is
	'PLyZd6Up2ag97WCkrF94SoTYpo9el5Jbl4', #256 - Justin Y. Encounter
	'PLEDLY9RcAVEAPgF9Oas2HY1Irv0fANt-v', #414 - Videos involving me
	'PLsLRmcu39w_EWxoKXVu3BP3_YT_xVhevT', #187 - Justin Y commented
	'PLcl0qJ8euYS2KNnNBAsl_AYpMGj559ql_', #386 - The chonicles of human achievement
	'PLEDLY9RcAVECo1fh3XI4Thh5yORYLcYu5', #71 - Stuff to watch at 3 AM
	'PL36thkaVLp_cjbfm7wC5Y72C0CKXclILj', #85 - Justin Y sightings
	'PLoJiUwNc7oHTyNVH5r-D357Quz27k6C2r', #87 - Every Video I've watched and found Justin Y. in it
	'PLWoMlj_1yvw7u4Swrh6xLSK62BfEi-clx', #101 - Justin Y, 03:00 69 (Oldest Published)
	'PL8cL6SlaG8o9sl93VEY-WvCdFlC1sW_Fb', #91 - Justin Y
	'PLzuaYEgVgFn6K3pxIkVGHx6reD61Gi4HH', #412 - Can You Find Justin Y. in the comments
	'PL73NEkiN4yYhmHWwxuLKoH4p-NdJess4p', #303 - Justin Y. will be obliterated
	'PLWtaAQnBZMhc7wMKeV5OYm-HgLw5FFhFo', #129 - Mystery
	'PLjbgElHjuOFt7Jr96LO44-hnA0mvk8OQN', #119 - Justin Y. Comments in videos
	'PLwDG7WIMSrIv5yyHK2iAe4NPaw7shSwA-', #74 - Videos I watch that Justin y commented on
	'PLlkgpk6VYN-VC06r47a-ujPy817WRItP8', #138 - Justin Y.
	'PLlDNv7LhQsqHNwz_qCLoQ00uV3ZaUZWKm', #207 - Justin Y
	'PL_KQOLf_bJf2MmCiYiWeXTPpZuh00yrTT', #94 - Justin Y.
	'PLWoMlj_1yvw6I2T8LBDRE3Dl-009QG0UV', #196 - Justin Y. Presents The Great Ones (Oldest Published)
	'PL-qupiaXIo35dwoLWPX4h21-AhCEghOJk', #127 - Videos with Justin Y Comments
	'PL2S1ygZEsbrq7LnFYZ4Vo_fwBAZN_usLl', #38 - Videos Justin Y comments on
	'PLWCtQoesywqPQhmKo6r8_za2gd6qzg89L', #40 - JUSTIN Y.
	'PLW1z7JmxIwi3NngIaihev50HbibVMUR7T', #108 - Justin T.
	'PLfnjQQeSRtRDJGvPY9_2x3bg1Ibd5UW3t', #43 - Justin y
	'PLqfTYcfGfk2Z95KaNly6gLV2ksDr6BW1i', #48 - Justin Y. Video playlist
	'PLmV_z0VTBoQM7I0c2P0Ne89mFy6GRRrNz', #150 - Justin Y. Is here
	'PLpZ6EQdG37vDF0-hslIe6BkcKxFwwvyYW', #1179 - Weird side of YouTube
	'PLEDLY9RcAVEClHaZTv4D2B_1ZbT6prKzR', #55 - Weeb Stuff
	'PLc5e3gLSmfQoKhG4TaujsyQ9Tnzo2HusM', #53 - Justin. Y was here
	'PL5IiKBxO9ciwo1kzKsTRWAttaUPOJa7ko', #69 - Justin Y
	'PL2P9kapzgxPKt1R4a1uhaSTrBBWgYliP3', #313 - Vids With Justin Y.'s Comment
	'PLaMsHFoQCfi-c04jfFe5tRbWPjphMrNs7', #65 - Videos I've seen Justin Y. in
	'PLTMsl9Zjn8658xhd3QolGv0xeQ5qtfjSq', #25 - Finding Justin y
	'PLMqyXJorJPzAvyHgV4rrixNuIEjI2Jc-g', #31 - Where I find Justin Y.
	'PLth-U9OH9GxXZf6f3rUFIyfne-eBV4cU8', #36 - Justin Y. hunt
	'PLg1UXes0H5Kr_0pirdTJSUPYtBSlYsPoq', #21 - Justin y is here
	'PLc6nYgH9n1HnVaiKaoEwJSywuoeewRC1B', #41 - The great ones
	'PLOJTuMA4-7oEAb6qTHG6J98Q66452tQyw', #35 - Nightcore
	'PL5as-6qnU5d87-bosb3Edbp1tRL-UYxTP', #2706 - The best Videos on YouTube and Vines
	'PLv3TTBr1W_9tppikBxAE_G6qjWdBljBHJ', #2799 - Instant Regret Clicking this Playlist
	'PLSCxT16tijxi7VuC3Uh3nPLmJocr7Ju28', #2346 - Cy's dank meme playlist
	'LLp1FjmTu8nw4lkKlU7iI74w',           #801 - Gavin's Liked Videos
	'LLa6TeYZ2DlueFRne5DyAnyg',           #910 - Grandayy's Liked Videos
	'LL1EW42tsTQTuFIkKcAhufkw',           #616 - Talking Nonsense's Liked Videos
	'LLq_X6pFQK2r8ptOq64nMYbA',           #5000 - Misaka Mikoto's Liked Videos
	'LLFBuuvyZWLmYX_ve0RsBxbQ',           #1261 - kermit's Liked Videos
	'LLQMjMW-9PhWoH6TWwmnVWvA',           #2762 - CallMeCarson's Liked Videos
	'LLt7E8Qpue2TU9Yh47vkEbsQ',           #420 - Dolan Dark's Liked Vidoes
	'LL0vXwnNFwrXRlje-gSxw-Eg',           #5000 - DatfaceddoeThe2nd Aka The Master Of Kirby's Liked Videos
	'LLt-GOpCw4dOBlIyqL9A1ztA',           #5000 - Sr Pelo's Liked Videos
	'LLMYTaTc_gVRyGF6LWzdIsqA',           #3551 - Cyranek's Liked Videos
	'LL9ecwl3FTG66jIKA9JRDtmg',           #3221 - SiIvaGunner's Liked Videos
	'LLYzPXprvl5Y-Sf0g4vX-m6g',           #1288 - jacksepticeye's Liked Videos
	'LLk6rHCnCNxqWHKFknnyZGZw',           #781 - blazeaster
	'PL68kEVQCeE3oxk3hZms2nJ0s3kYrp0rbj', #4873 - Slightly Less Important Videos I
	'LLo8bcnLyZH8tBIH9V1mLgqQ',           #1145 - TheOdd1sOut's Liked Videos
	'LLllm3HivMERwu2x2Sjz5EIg',           #795 - Vargskelethor Joel's Liked Videos
	'LLny_vGt2N7_QJ5qBOAHxlcw',           #932 - maxmoefoe's Liked Videos
	'LLQ4FyiI_1mWI2AtLS5ChdPQ',           #1163 - Boyinaband's Liked Videos
	'PLv3TTBr1W_9vPB6WPEnPwOpLYeZQW5tuD', #4998 - Instant Regret Clicking This Playlist 2.0
	'LLGwu0nbY2wSkW8N-cghnLpA',           #818 - Jaiden Animations's Liked Videos
	'LLPcFg7aBbaVzXoIKSNqwaww',           #1451 - jacksfilms's Liked Videos
	'LLu6v4AdYxVH5fhfq9mi5llA',           #459 - Let Me Explain Studios's Liked Videos
	'LLJ0-OtVpF0wOKEqT2Z1HEtA',           #318 - ElectroBOOM's Liked Videos
	'LLo1qj9072AgkWlmkR-PLwCQ',           #486 - AngeloJFurfaro's Liked Videos
]

#Create directory to hold downloaded data
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

#Try to load the existing database so far
all_comments = {}
try:
	with codecs.open(DATA_FILE, 'r', encoding='utf-8') as fin:
		for line in fin:
			vid, title, comment = line[:-1].split('~')
			all_comments[vid] = (title, comment)
except:
	pass

def scrape_playlist(playlist):
	NEXT_PAGE = ''
	while True:
		#Setup strings
		PLAYLIST_URL = 'https://www.googleapis.com/youtube/v3/playlistItems?key=' + API_KEY + '&part=snippet&playlistId=' + playlist + '&maxResults=50'
		if NEXT_PAGE != '':
			PLAYLIST_URL += '&pageToken=' + NEXT_PAGE
		SAVE_FILE = SAVE_DIR + 'ply_' + playlist + '_' + str(NEXT_PAGE) + '.txt'
		data_out = codecs.open(DATA_FILE, 'a', encoding='utf-8')

		#Download the query (or load from file if cached)
		if os.path.isfile(SAVE_FILE):
			query_str = ""
			with codecs.open(SAVE_FILE, 'r', encoding='utf-8') as fin:
				query_str = fin.read()
		else:
			if READ_CACHE_ONLY:
				break
			query_str = urlopen(PLAYLIST_URL, timeout=10).read().decode('utf-8')
			with codecs.open(SAVE_FILE, 'w', encoding='utf-8') as fout:
				fout.write(query_str)

		#Ignore if query is empty
		if len(query_str) == 0:
			print("===== WARNING: Empty Response =====")
			return

		#Loop over all videos in the playlist
		query_json = json.loads(query_str)
		items = query_json['items']
		for item in items:
			#Get the video information
			snippet = item['snippet']
			title = snippet['title']
			vid = snippet['resourceId']['videoId']
			if vid in all_comments:
				continue

			#Scrape Justin Y comments from the video
			try:
				good_comments = scrape_api(vid)
			except:
				good_comments = []

			#Clean the text a bit and add it to the data set
			title = title.replace('\n',' . ').replace('\r',' . ').replace('~',' ')
			print(title.encode('utf-8').decode())

			#Save all the good comments
			for good_comment in good_comments:
				good_comment = good_comment.replace('\n',' . ').replace('\r',' . ').replace('~',' ')
				data_out.write(vid + '~' + title + '~' + good_comment + '\n')
				all_comments[vid] = (title, good_comment)
				print("    " + good_comment.encode('utf-8').decode())

		#Get the next page to process or quit if done
		data_out.close()
		if 'nextPageToken' in query_json:
			NEXT_PAGE = query_json['nextPageToken']
		else:
			break

def scrape_api(vid):
	cur_page = ''
	good_comments = []
	for i in range(NUM_COMMENT_PAGES):
		#Setup strings
		COMMENT_URL = 'https://www.googleapis.com/youtube/v3/commentThreads?key=' + API_KEY + '&textFormat=plainText&part=snippet&videoId=' + vid + '&maxResults=100&order=relevance'
		if len(cur_page) > 0:
			COMMENT_URL += '&pageToken=' + cur_page
		SAVE_FILE = SAVE_DIR + 'com_' + vid + cur_page + '.txt'

		#Download the query (or load from file if cached)
		if os.path.isfile(SAVE_FILE):
			query_str = ''
			with codecs.open(SAVE_FILE, 'r', encoding='utf-8') as fin:
				query_str = fin.read()
		else:
			if READ_CACHE_ONLY:
				continue
			query_str = urlopen(COMMENT_URL).read().decode('utf-8')
			with open(SAVE_FILE, 'w', encoding='utf-8') as fout:
				fout.write(query_str)

		query_json = json.loads(query_str)
		items = query_json['items']
		if ALL_COMMENTS:
			#Look for popular comments and add them
			for j in range(len(items)):
				item = items[j]
				snippet = item['snippet']['topLevelComment']['snippet']
				num_likes = int(snippet['likeCount'])
				if num_likes < 50 or (j >= 3 and num_likes < 200):
					continue
				comment = snippet['textDisplay']
				if len(comment) < ALL_COMMENTS_LEN_MIN or len(comment) > ALL_COMMENTS_LEN_MAX:
					continue
				good_comments.append(comment)
		else:
			#Look for Justin Y comments
			justin_y_comment = ''
			for item in items:
				snippet = item['snippet']['topLevelComment']['snippet']
				if JUSTIN_Y_CHANNEL in snippet['authorChannelUrl']:
					justin_y_comment = snippet['textDisplay']
					break

			#Return result if found
			if len(justin_y_comment) > 0:
				return [justin_y_comment]

		#Otherwise search the next page
		if 'nextPageToken' in query_json:
			cur_page = query_json['nextPageToken']
		else:
			break

	#Return whatever was found
	return good_comments

for playlist in ALL_PLAYLISTS:
	print("==========================================================")
	print("     Staring playlist " + playlist)
	print("==========================================================")
	print("")
	scrape_playlist(playlist)

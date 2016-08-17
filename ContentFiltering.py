from __future__ import division
import re
import sys
import math
import random
import numpy as np
import xml.etree.ElementTree as ET


def tokenize(text, word_re, splitter = None):
    if text is None or text == 'N/A': 
        return []

    if splitter is None: 
        ret = [word_re.search(w).group(0) for w in text.split() if word_re.search(w)]
        return ret
    ret = [word_re.search(w).group(0) for w in text.split(splitter) if word_re.search(w)]
    return ret

def normalize(tvShows):
    if len(tvShows) == 0: 
        return []
    ret = [show.strip().lower() for show in tvShows]
    return ret

def encode(text):
    if len(text) == 0: 
        return []
    ret = [w.encode('utf-8') for w in text]
    return ret
    
def removeStopWords(text, stopwords):
    ret = [w for w in text if not w in stopwords]
    return ret

def filter(regex, text):
    ret = [w for w in text if not regex.match(w)]
    return ret

def addTvShows(TV_SHOWS, idSeq, LIKES, userId, tvShows):
    showIds = set()
    for show in tvShows:
        if show not in TV_SHOWS:
            idSeq = idSeq + 1
            TV_SHOWS[show] = idSeq
            showIds.add(idSeq)
        else:
            showIds.add(TV_SHOWS[show])
    LIKES[userId] = showIds
    return idSeq


def getSelectedGenres(stopWords):
    tv_genre = set()
    with open('data/shows_all_stemmed.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split('\t')
            tv_genre = tv_genre.union( removeStopWords(l[1].split(), stopWords) )
    return tv_genre

def getShowsToGenres():
    shows = dict()
    with open('data/shows_all_stemmed.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split('\t')
            tv_genre = tv_genre.union( removeStopWords(l[1].split(), stopWords) )
    return tv_genre

def getProbabilityForUser(user):
    count = dict()
    with open('data/shows_all.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split('\t')
            count[l[0].strip()]=l[1].split(',')
    # print count
    return count


def sample(all, fraction):
    return random.sample(all, int(math.ceil(fraction * len(all))))

def createUserVector(user_words, vocabulary, vocab_size):
    result = [0] * vocab_size
    for w in user_words:
        result[ vocabulary[w] ] += 1
    return result

def cosine2(a, b):
    if len(a) != len(b):
        raise ValueError, "a and b must be same length"
    numerator = 0
    denoma = 0
    denomb = 0
    for i in range(len(a)):
        ai = a[i]             
        bi = b[i]
        numerator += ai*bi    
        denoma += ai*ai
        denomb += bi*bi
    result = 1 - numerator / (math.sqrt(denoma * denomb))
    return result


def cosine(u1, u2):
    norm1 = np.linalg.norm(u1)
    norm2 = np.linalg.norm(u2)
    ret = np.dot(u1, u2) / ( norm1 * norm2 )
    return ret

def evaluate(user, LIKES, similar_users):
    shows_recommended = set()
    for sim_user in similar_users:
        shows_recommended.update(LIKES[sim_user])
    if len(shows_recommended) == 0: 
        return (0, 0)
    precision = len(shows_recommended.intersection(LIKES[user])) / len(shows_recommended)
    recall = len(shows_recommended.intersection(LIKES[user])) / len(LIKES[user])
    return precision, recall

def computeSimilarity(USER_VECTORS, tv_watchers, test_set):
    control_set = tv_watchers - test_set
    sim_matrix = dict()
    for t in test_set:
        scores = dict() 
        for c in control_set:
            scores[c] = cosine(USER_VECTORS[t], USER_VECTORS[c])
        sim_matrix[t] = sorted(scores, key=scores.get, reverse=True)[:10]
    print sim_matrix
    return sim_matrix            
                                               
def parse(fbDataFile, stopwords):
    word_re = re.compile('\w+') 
    TV_SHOWS = {}
    LIKES = {}
    FEATURES = {}
    showId = 0
    tree = ET.parse(fbDataFile)
    root = tree.getroot()
    for user in root.findall('user'):
        userId = None
        tvShows = None
        about = None
        movies = None
        music = None
        books = None
        activities = None
        interests = None
        try:
            userId = user.attrib['id']
            tvShows = removeStopWords( encode( normalize( tokenize(user.find('tv').text, word_re, ',') ) ), stopwords )
            movies = removeStopWords( encode( normalize( tokenize(user.find('movies').text, word_re, ',') ) ), stopwords )
            music = removeStopWords( encode( normalize( tokenize(user.find('music').text, word_re, ',') ) ), stopwords )
            books = removeStopWords( encode( normalize( tokenize(user.find('books').text, word_re, ',') ) ), stopwords )
            interests = removeStopWords( encode( normalize( tokenize(user.find('interests').text, word_re, ',') ) ), stopwords )
            if user.find('activities') is not None:
                activities = removeStopWords( encode( normalize( tokenize(user.find('activities').text, word_re, ',') ) ), stopwords )            
            about = removeStopWords( encode( normalize( tokenize(user.find('about').text, word_re)) ), stopwords )
        except UnicodeEncodeError as uerr:
            pass
        showId = addTvShows(TV_SHOWS, showId, LIKES, userId, tvShows)
        FEATURES[userId] = about
        FEATURES[userId].extend(user.find('gender').text.split())
        FEATURES[userId].extend(user.find('locale').text.split())
        FEATURES[userId].extend(movies)
        FEATURES[userId].extend(books)
        FEATURES[userId].extend(music)
        FEATURES[userId].extend(interests)
        if activities: 
            FEATURES[userId].extend(activities)
    return TV_SHOWS, LIKES, FEATURES
        
stopwords = set( open('data/stop_words.txt', 'r').read().strip().split(',') )
TV_SHOWS, LIKES, FEATURES = parse("data/aggregate_data.xml", stopwords)
#
numbers = re.compile(r'[_\d.]+') # numbers and other strange tokens made up of underscores; re.compile(r'[\d.]*\d+')
# Parse TV_SHOWS genre text
tv_titles = []
tv_genre = []
tv_text = []
with open('data/shows_all_stemmed.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        l = line.strip().split('\t')
        tv_titles.append( removeStopWords(l[0].split(), stopwords) )
        tv_genre.append( removeStopWords(l[1].split(), stopwords) )
        tv_text.append( removeStopWords(l[2].split(), stopwords) )
tv_watchers = set()
CURATED_LIKES = dict()
CURATED_SHOWS = dict()
curated_show_id = 1
inverted = dict( (id, show) for (show, id) in TV_SHOWS.iteritems() )
for uid in LIKES.keys():
    userCuratedShows = set()
    for showId in LIKES[uid]:
        show_title = inverted[showId].split()
        FEATURES[uid].extend(show_title)
        print showId
        for idx, title in enumerate(tv_titles):
            if len(set(show_title).intersection(title)) > 0:
                tv_watchers.add(uid)
                key = ''.join(title)
                if not key in CURATED_SHOWS:
                    CURATED_SHOWS[key] = curated_show_id
                    userCuratedShows.add(curated_show_id)
                    curated_show_id += 1
                else:
                    userCuratedShows.add(CURATED_SHOWS[key])
                print title, key, CURATED_SHOWS[key]
                FEATURES[uid].extend([w for w in sum([genre.split(',') for genre in tv_genre[idx]], []) if len(w) > 0])
    FEATURES[uid] = filter(numbers, FEATURES[uid])
    CURATED_LIKES[uid] = userCuratedShows


all_features = set([])
for uid in FEATURES.keys(): 
    print uid, all_features
    all_features.update(FEATURES[uid])
all_words = sorted(list(all_features))
vocabulary = dict()
for idx, word in enumerate(all_words):
    vocabulary[word] = idx

print vocabulary
USER_VECTORS = dict()
for uid in FEATURES.keys():
    USER_VECTORS[uid] = createUserVector(FEATURES[uid], vocabulary, len(all_words))
    print USER_VECTORS[uid]
test_set = set(sample(tv_watchers, 0.3))
sim_matrix = computeSimilarity(USER_VECTORS, tv_watchers, test_set)
aggr_precision = 0.0
aggr_recall = 0.0
for u in sim_matrix.keys():
    (precision, recall) = evaluate(u, CURATED_LIKES, sim_matrix[u])
    print precision, recall
    aggr_precision += precision
    aggr_recall += recall
aggr_size = len(sim_matrix.keys())
P = aggr_precision/aggr_size
R = aggr_recall/aggr_size
print P, R



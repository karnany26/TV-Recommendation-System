import re
import sys
import math
import random
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords 



def tokenize(tvShows, splitter = None):
    if tvShows is None or tvShows == 'N/A': return []
    if splitter is None: return tvShows.split()
    return tvShows.split(splitter)

def normalize(tvShows):
    if len(tvShows) == 0: return []
    return [show.strip().lower() for show in tvShows]


def encode(text):
    if len(text) == 0: return []
    return [w.encode('utf-8') for w in text]

def removeStopWords(text, Stopwords):
    words = set()
    for w in text:
        if not w in Stopwords:
            words.add(w)
    # [words.union(w) for w in text if not w in Stopwords]
    # print words
    return words

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

def addItems(LIKES):
    LIKED_BY = {}
    for u in LIKES.keys():
        for show in LIKES[u]:
            if not show in LIKED_BY: LIKED_BY[show] = set([])
            LIKED_BY[show].add(u)
    return LIKED_BY

def parse(fbDataFile, UserUser=True):
    TV_SHOWS = {}
    LIKES = {}
    showId = 0
    tree = ET.parse(fbDataFile)
    root = tree.getroot()
    for user in root.findall('user'):
        userId = None
        tvShows = None
        about = None
        try:
            userId = user.attrib['id']
            tvShows = encode( normalize( tokenize(user.find('tv').text, ',') ) )
        except UnicodeEncodeError as uerr:
            pass
        showId = addTvShows(TV_SHOWS, showId, LIKES, userId, tvShows)
    if UserUser: return TV_SHOWS, LIKES
    # Item-Based
    LIKED_BY = addItems(LIKES)
    return TV_SHOWS, LIKED_BY

def sample(all, fraction):
    return random.sample(all, int(math.ceil(fraction * len(all))))


def sampleDict(all, fraction):
    numberOfUsers = int(math.ceil(fraction * len(all)))
    count = 0
    sampled = {}
    for key in all:
        if count>= numberOfUsers:
            break
        else:
            sampled[key] = all[key]
            count=count+1
    return sampled


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
    with open('data/shows_all.txt', 'r') as f:
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



stopWords = stopwords.words("english")
stopwords1 = set( open('data/stop_words.txt', 'r').read().strip().split(',') )
stopWords = set(stopWords).union(stopwords1)


TV_SHOWS, LIKES = parse("data/aggregate_data_stemmed.xml") #, False)    

total_precision = 0.0
total_recall = 0.0



tv_titles = []
tv_genre = []
tv_text = []
with open('data/shows_all_stemmed.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        l = line.strip().split('\t')
        tv_titles.append( removeStopWords(l[0].split(), stopWords) )
        tv_genre.append( removeStopWords(l[1].replace(',',' ').split(), stopWords) )
        tv_text.append( removeStopWords(l[2].split(), stopWords) )

tv_watchers = set() 
CURATED_LIKES = dict()
CURATED_SHOWS = dict()
CURATED_GENRES = dict()
curated_show_id = 1
LIKES = sampleDict(LIKES,1)
print len(LIKES)
inverted = dict( (id, show) for (show, id) in TV_SHOWS.iteritems() )
for uid in LIKES.keys():
    userCuratedShows = set()
    for showId in LIKES[uid]:
        show_title = inverted[showId].split()
        for idx, title in enumerate(tv_titles):
            if len(set(show_title).intersection(title)) > 0:
                tv_watchers.add(uid)
                key = ''.join(title)
                if not key in CURATED_SHOWS:
                    CURATED_SHOWS[key] = curated_show_id
                    curated_genre_key = []
                    for w in tv_genre[idx]:
                        curated_genre_key.append(w)
                    # print "============curated_genre_key================"
                    # print curated_genre_key
                    CURATED_GENRES[key] = curated_genre_key
                    userCuratedShows.add(curated_show_id)
                    curated_show_id += 1
                else:
                    userCuratedShows.add(CURATED_SHOWS[key])
                # The 'sum' trick below is used to flatten out the list: sum(l, [])
                # will flatten a list of lists
    CURATED_LIKES[uid] = userCuratedShows


print CURATED_SHOWS
print len(CURATED_SHOWS)
CURATED_LIKES_NEW = {}
for (x,y) in CURATED_LIKES.iteritems():
    if(len(y)>5):
        CURATED_LIKES_NEW[x] = y
CURATED_LIKES = CURATED_LIKES_NEW


INVERSE_CURATED_SHOWS = dict( (id, show) for (show, id) in CURATED_SHOWS.iteritems() )

USER_GENRE_SCORE = {}

CURATED_LIKES_TRAIN = {}
#training
for user in CURATED_LIKES:
    CURATED_LIKES_TRAIN[user] = set(random.sample(CURATED_LIKES[user], int(math.ceil(0.5 * len(CURATED_LIKES[user])))))
    genres = {}
    for show in CURATED_LIKES_TRAIN[user]:
        for genre in CURATED_GENRES[INVERSE_CURATED_SHOWS[show]]:
            if genre in genres:
                genres[genre] = genres[genre] + 1
            else:
                genres[genre] = 1

    USER_GENRE_SCORE[user] = genres

for user in USER_GENRE_SCORE:
    sum = 0.0
    for genre in USER_GENRE_SCORE[user]:
        sum+=USER_GENRE_SCORE[user][genre]
    for genre in USER_GENRE_SCORE[user]:
        USER_GENRE_SCORE[user][genre] = USER_GENRE_SCORE[user][genre]/sum


    
RECOMMENDATIONS = {}
CURATED_LIKES_TEST = {}
for user in USER_GENRE_SCORE:
    CURATED_LIKES_TEST[user] = CURATED_LIKES[user].difference(CURATED_LIKES_TRAIN[user])
    SHOW_SCORES = {}
    for show in CURATED_SHOWS:
        score = 0.0
        for genre in CURATED_GENRES[show]:
            if genre in USER_GENRE_SCORE[user]:
                score += USER_GENRE_SCORE[user][genre]
        SHOW_SCORES[show] = score
    recos = sorted([(k,v) for (k, v) in SHOW_SCORES.iteritems() if v > 0.0], key=lambda tup: -tup[1])
    recos = [x for (x,y) in recos]
    # recos = sorted(SHOW_SCORES.iteritems(), key=lambda (x,y):float(-y))
    numRecos = 15*len(CURATED_LIKES_TEST[user])
    RECOMMENDATIONS[user] = set(recos[:20])
    testing_shows = set([INVERSE_CURATED_SHOWS[id] for id in CURATED_LIKES_TEST[user]])
    print "--------------------------testing_shows-----------------"
    print testing_shows
    print "--------------------------RECOMMENDATIONS---------------"
    print RECOMMENDATIONS[user]
    precision = float(len(RECOMMENDATIONS[user].intersection(testing_shows))) / len(RECOMMENDATIONS[user])
    recall = float(len(RECOMMENDATIONS[user].intersection(testing_shows))) / len(testing_shows)
    total_precision += precision
    total_recall += recall

total_precision = total_precision/len(USER_GENRE_SCORE)
print total_precision
total_recall = total_recall/len(USER_GENRE_SCORE)
print total_recall
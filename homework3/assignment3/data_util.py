#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter

import numpy as np
from util import read_conll, one_hot, window_iterator, ConfusionMatrix, load_word_vector_mapping
from defs import LBLS, NONE, LMAP, NUM, UNK, EMBED_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"

def casing(word):
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def featurize(embeddings, word):
    """
    Featurize a word given embeddings.
    """
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))

def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        self.max_length = max_length

    def vectorize_example(self, sentence, labels=None):
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]] for word in sentence]
        if labels:
            labels_ = [LBLS.index(l) for l in labels]
            return sentence_, labels_
        else:
            return sentence_, [LBLS[-1] for _ in sentence]

    def vectorize(self, data):
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]

    @classmethod
    def build(cls, data):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=10000)
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        #{'CASE:aa': 2, 'CASE:AA': 3, 'CASE:Aa': 4, 'CASE:aA': 5}
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        max_length = max(len(sentence) for sentence, _ in data)

        return cls(tok2id, max_length)

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)

def load_and_preprocess_data(args):
    logger.info("Loading training data...")
    train = read_conll(args.data_train)
    # ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'] ['ORG', 'O', 'MISC', 'O', 'O', 'O', 'MISC', 'O', 'O']
    # ['Peter', 'Blackburn'] ['PER', 'PER']
    # ['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']
    logger.info("Done. Read %d sentences", len(train))
    logger.info("Loading dev data...")
    dev = read_conll(args.data_dev)
    logger.info("Done. Read %d sentences", len(dev))

    helper = ModelHelper.build(train)

    # now process all the input data.
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)
    """
    Let see an example: args.data_train: file tiny.conll
    train = train[0:5]
    The result of train
    [(['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ['ORG', 'O', 'MISC', 'O', 'O', 'O', 'MISC', 'O', 'O']), 
    (['Peter', 'Blackburn'], ['PER', 'PER']),
     (['BRUSSELS', '1996-08-22'], ['LOC', 'O']), 
     (['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad', 'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.'], ['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'MISC', 'O', 'O', 'O', 'O', 'O', 'MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']), 
     (['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.'], ['LOC', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'O', 'O', 'O', 'PER', 'PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])]

    The result of helper.tok2id

    {'CASE:aA': 56, 'CASE:AA': 57, '1996-08-22': 14, 'committee': 15, 'be': 38, 'german': 4, 'it': 16, 'boycott': 17, 
    'britain': 18, 'werner': 19, 'determine': 20, 'UUUNKKK': 62, 'eu': 21, 'peter': 22, '</s>': 61, 'lamb': 5, 'disagreed': 24, 
    'said': 6, 'from': 25, 'sheepmeat': 26, 'consumers': 7, 'rejects': 27, 'union': 28, 'veterinary': 29, 'thursday': 30, '.': 2, 
    'zwingmann': 31, 'to': 1, 'other': 33, 'call': 34, 'scientists': 35, 'was': 36, 'until': 8, 'european': 9, 'CASE:aa': 59, 'sheep': 23, 
    'buy': 39, "'s": 10, 'scientific': 37, 'advice': 11, 'clearer': 40, 'wednesday': 41, 'germany': 42, 'mad': 43, 'shun': 44, 'brussels': 45, 
    'with': 46, 'than': 47, 'on': 12, 'disease': 53, 'cow': 48, 'blackburn': 49, 'whether': 50, 'should': 51, 'countries': 52, 'commission': 32, 
    'british': 13, 'CASE:Aa': 58, '<s>': 60, 'transmitted': 54, 'can': 55, 'the': 3, 'representative': 56}

    The result of train_data
    [([[21, 57], [27, 59], [4, 58], [34, 59], [1, 59], [17, 59], [13, 58], [5, 59], [2, 56]], [1, 4, 3, 4, 4, 4, 3, 4, 4]), 
    ([[22, 58], [49, 58]], [0, 0]), 
    ([[45, 57], [14, 56]], [2, 4]), 
    ([[3, 58], [9, 58], [32, 58], [6, 59], [12, 59], [30, 58], [16, 59], [24, 59], [46, 59], [4, 58], [11, 59], [1, 59], [7, 59], [1, 59], [44, 59], [13, 58], [5, 59], [8, 59], [35, 59], [20, 59], [50, 59], [43, 59], [48, 59], [53, 59], [55, 59], [38, 59], [54, 59], [1, 59], [23, 59], [2, 56]], [4, 1, 1, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]), 
    ([[42, 58], [10, 59], [56, 59], [1, 59], [3, 59], [9, 58], [28, 58], [10, 59], [29, 59], [15, 59], [19, 58], [31, 58], [6, 59], [12, 59], [41, 58], [7, 59], [51, 59], [39, 59], [26, 59], [25, 59], [52, 59], [33, 59], [47, 59], [18, 58], [8, 59], [3, 59], [37, 59], [11, 59], [36, 59], [40, 59], [2, 56]], [2, 4, 4, 4, 4, 1, 1, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4])]

    """


    return helper, train_data, dev_data, train, dev

def load_embeddings(args, helper):
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(args.vocab, args.vectors).items():
        word = normalize(word)
        if word in helper.tok2id:
            embeddings[helper.tok2id[word]] = vec
    logger.info("Initialized embeddings.")

    return embeddings

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}


def get_chunks(seq, default=LBLS.index(NONE)):
    """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def test_get_chunks():
    assert get_chunks([4, 4, 4, 0, 0, 4, 1, 2, 4, 3], 4) == [(0,3,5), (1, 6, 7), (2, 7, 8), (3,9,10)]

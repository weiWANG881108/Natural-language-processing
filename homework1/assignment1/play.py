#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from utils.treebank import StanfordSentiment
import utils.glove as glove

from q3_sgd import load_saved_params, sgd

# We will use sklearn here because it will run faster than implementing
# ourselves. However, for other parts of this assignment you must implement
# the functions yourself!
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def getArguments():
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--pretrained", dest="pretrained", action="store_true",
		help="Use pretrained GloVe vectors.")
	group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
		help="Use your vectors from q3.")
	return parser.parse_args()

def getSentenceFeatures(tokens, wordVectors, sentence):
	"""
	Obtain the sentence feature for sentiment analysis by averaging its
	word vectors
	"""

	# Implement computation for the sentence features given a sentence.

	# Inputs:
	# tokens -- a dictionary that maps words to their indices in
	#           the word vector list
	# wordVectors -- word vectors (each row) for all tokens
	# sentence -- a list of words in the sentence of interest

	# Output:
	# - sentVector: feature vector for the sentence

	sentVector = np.zeros((wordVectors.shape[1],))

	### YOUR CODE HERE
	for word in sentence:
	    sentVector += wordVectors[tokens[word]].reshape(wordVectors.shape[1],)
	### END YOUR CODE

	assert sentVector.shape == (wordVectors.shape[1],)
	return sentVector

def getRegularizationValues():
	"""Try different regularizations

	Return a sorted list of values to try.
	"""
	values = None   # Assign a list of floats in the block below
	### YOUR CODE HERE
	values = [10**i for i in np.random.uniform(-5,0,10)]
	### END YOUR CODE
	return sorted(values)
	

def main(args):
	dataset = StanfordSentiment()
	tokens = dataset.tokens()
	nWords = len(tokens)

	if args.yourvectors:
		_, wordVectors, _ = load_saved_params()
		wordVectors = np.concatenate(
			(wordVectors[:nWords,:], wordVectors[nWords:,:]),
			axis=1)
	elif args.pretrained:
		wordVectors = glove.loadWordVectors(tokens)
	dimVectors = wordVectors.shape[1]
	print dimVectors

	trainset = dataset.getTrainSentences()
	nTrain = len(trainset)
	trainFeatures = np.zeros((nTrain, dimVectors))
	trainLabels = np.zeros((nTrain,), dtype=np.int32)
	for i in xrange(nTrain):
		words, trainLabels[i] = trainset[i]
		trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

	# Prepare dev set features
	devset = dataset.getDevSentences()
	nDev = len(devset)
	devFeatures = np.zeros((nDev, dimVectors))
	devLabels = np.zeros((nDev,), dtype=np.int32)
	for i in xrange(nDev):
		words, devLabels[i] = devset[i]
		devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

	# Prepare test set features
	testset = dataset.getTestSentences()
	nTest = len(testset)
	testFeatures = np.zeros((nTest, dimVectors))
	testLabels = np.zeros((nTest,), dtype=np.int32)
	for i in xrange(nTest):
		words, testLabels[i] = testset[i]
		testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)


if __name__ == "__main__":
	main(getArguments())

import csv
import csv_helper as ch
from collections import defaultdict
import operator 
import cPickle
from itertools import islice
import sklearn.svm
from sklearn import svm
import numpy as np

# TRAINING_INPUT_FILE = 'data/positive_negative_reviews_sentiment_2k.csv'
TRAINING_INPUT_FILE = 'data/2k/output_2k.csv'
ngramDict = []
sentenceNgramDict = []

def main():
	global ngramDict

	ngramsTrimmed = getNGramsFromTrigramData()
	# print len(ngramsTrimmed)
	# print len(ngramsTrimmed[0])
	# print len(ngramsTrimmed[1])
	# print len(ngramsTrimmed[2])
	# print ngramsTrimmed[0][:30]
	# print ngramsTrimmed[1][:30]
	# print ngramsTrimmed[2][:30]
	# return

	ngrams = getNGramsFromTrainingData()
	populateSentenceNgramDict(ngramsTrimmed)
	# print len(ngrams[0])
	# print len(ngrams[1])
	# print len(ngrams[2])

	# with open(r"ngrams_v2.pickle", "wb") as output_file:
	# 	cPickle.dump(ngrams, output_file)

	# with open(r"ngrams.pickle", "rb") as input_file:
	# 	ngrams = cPickle.load(input_file)

	print len(ngrams[0])
	ngrams[0] = ngrams[0][:10000]
	# ngrams[0] = ngrams[0][:1000]
	print len(ngrams[0])
	print len(ngrams[1])
	print len(ngrams[2])
	print ngrams[0][:10]
	print ngrams[1][:10]
	print ngrams[2][:10]
	# return
	populateNgramDict(ngrams)
	featureTable = buildFeatureTable()
	featureVectors = featureTable[0]
	labels = featureTable[1]
	x = featureVectors
	y = labels

	# print len(x)
	# print len(y)
	# return

	n = len(x)
	
	svc = svm.LinearSVC(C=1)
	scores = []
	# print y[:5]
	# return
	for i in range(10):

		start = i * (n/10)
		end = start + (n/10)
		accuracy = svc.fit(x[:start] + x[end:],y[:start] + y[end:]).score(x[start:end],y[start:end])
		# accuracy = svc.fit(x[1000:],y[1000:]).score(x[:1000],y[:1000])
		scores.append(accuracy)
		print accuracy
	
	print np.mean(np.array(scores))
	

	return

def populateSentenceNgramDict(ngrams):
	global sentenceNgramDict
	n = len(ngrams)
	sentenceNgramDict = [{} for i in range(n)]
	for i in range(n):
		for j,word in enumerate(ngrams[i]):
			sentenceNgramDict[i][word] = j

def populateNgramDict(ngrams):
	global ngramDict
	n = len(ngrams)
	ngramDict = [{} for i in range(n)]
	for i in range(n):
		for j,word in enumerate(ngrams[i]):
			ngramDict[i][word] = j


def buildFeatureTable():
	global TRAINING_INPUT_FILE
	global ngramDict
	global sentenceNgramDict 
	length = [len(x) for x in ngramDict]
	l1 = sum(length)
	lengthForSentenceTrigrams = [len(x) for x in sentenceNgramDict]
	l2 = sum(lengthForSentenceTrigrams)
	# print sum(length)
	# print sum(lengthForSentenceTrigrams)
	# print len(sentenceNgramDict[0])
	# print len(sentenceNgramDict[1])
	# print len(sentenceNgramDict[2])
	# print 'sdfsdfdsfsd'
	# return
	featureVectors = []
	labels = []
	with open(TRAINING_INPUT_FILE, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i,row in enumerate(reader):
			if i == 0:
				continue
			if i%250 == 0:
				# print row[-1]
				print i
			# instance = [0] * (sum(length))
			instance = [0] * (l1 + l2)
			body = row[2]
			prev = [''] * 2
			for word in body.split():
				# instance[-1] = row[-1]
				# labels.append(row[-1])
				unigram = word
				bigram = prev[1] + ' ' + unigram
				trigram = prev[0] + ' ' + bigram

				if unigram in ngramDict[0]:
					instance[ngramDict[0][unigram]] += 1

				if bigram in ngramDict[1]:
					instance[ngramDict[1][bigram]] += 1

				if trigram in ngramDict[2]:
					instance[ngramDict[2][trigram]] += 1

				if unigram in sentenceNgramDict[0]:
					instance[l1+sentenceNgramDict[0][unigram]] += 1

				if bigram in sentenceNgramDict[1]:
					instance[l1+sentenceNgramDict[1][bigram]] += 1

				if trigram in sentenceNgramDict[2]:
					instance[l1+sentenceNgramDict[2][trigram]] += 1

				prev[0] = prev[1]
				prev[1] = word

			featureVectors.append(instance)
			labels.append(0 if row[-2] == 'negative' else 1)

	return [featureVectors,labels]

def getNGramsFromTrainingData(n = 3):
	global TRAINING_INPUT_FILE
	import nltk
	from nltk.corpus import stopwords

	ngrams = [defaultdict(int) for i in range(n)]
	print ngrams
	# return
	prev = [''] * (n-1)
	with open(TRAINING_INPUT_FILE, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i,row in enumerate(reader):
			body = row[2]

			for word in body.split():
				unigram = word
				bigram = prev[1] + ' ' + unigram
				trigram = prev[0] + ' ' + bigram
				# print 'unigram  = ' + unigram + ', bigram = ' + bigram

				ngrams[0][unigram] += 1
				ngrams[1][bigram] += 1
				ngrams[2][trigram] += 1

				prev[0] = prev[1]
				prev[1] = word

			if i > 20000000000:
				break
	print 'printing now'
	print len(ngrams[0])
	print len(ngrams[1])
	print len(ngrams[2])
	# print ngrams[1]

	stops = set(stopwords.words('english'))


	topNGramsWithFrequencies = [sorted(i.items(), key = operator.itemgetter(1), reverse = True) for i in ngrams]
	# topNGrams = [[y[0] for y in x] for x in topNGramsWithFrequencies]
	topNGrams = []
	for x in topNGramsWithFrequencies:
		array = []
		for y in x:
			array.append(y[0])

		topNGrams.append(array)

	topNGrams = [[y  for y in x if y.lower() not in stops] for x in topNGrams]
	print topNGrams[0][:50]
	print topNGrams[1][:50]
	print topNGrams[2][:50]
	print 
	print len(topNGrams[0])
	print len(topNGrams[1])
	print len(topNGrams[2])
	# topNGrams[0] = topNGrams[0][:10]
	topNGrams[1] = topNGrams[1][:10000]
	topNGrams[2] = topNGrams[2][:10000]

	sw = stopwords.words('english')
	if 'the' in sw:
		print 'found'
	else:
		print 'not found' 

	return topNGrams

def getNGramsFromTrigramData(n = 3):
	global TRAINING_INPUT_FILE
	import nltk
	from nltk.corpus import stopwords

	ngrams = [defaultdict(int) for i in range(n)]
	print ngrams
	# return
	prev = [''] * (n-1)
	print ch.getHeader(TRAINING_INPUT_FILE)
	print '**********************************'
	with open(TRAINING_INPUT_FILE, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i,row in enumerate(reader):
			# print len(row)
			trigramData = row[6]

			for sentenceTrigram in trigramData.split(','):
				sentenceTrigram = sentenceTrigram.split()
				unigram = sentenceTrigram[0] if len(sentenceTrigram) >= 1 else None
				bigram = ' '.join(sentenceTrigram[:2]) if len(sentenceTrigram) >= 2 else None
				trigram = ' '.join(sentenceTrigram[:3]) if len(sentenceTrigram) >= 3 else None
				# print 'unigram  = ' + unigram + ', bigram = ' + bigram

				if unigram is not None:
					ngrams[0][unigram] += 1
				if bigram is not None:	
					ngrams[1][bigram] += 1
				if trigram is not None:
					ngrams[2][trigram] += 1

				# prev[0] = prev[1]
				# prev[1] = word

			if i > 20000000000:
				break
	print 'printing now'
	print len(ngrams[0])
	print len(ngrams[1])
	print len(ngrams[2])
	# print ngrams[1]

	stops = set(stopwords.words('english'))


	topNGramsWithFrequencies = [sorted(i.items(), key = operator.itemgetter(1), reverse = True) for i in ngrams]
	# topNGrams = [[y[0] for y in x] for x in topNGramsWithFrequencies]
	topNGrams = []
	for x in topNGramsWithFrequencies:
		array = []
		for y in x:
			array.append(y[0])

		topNGrams.append(array)

	topNGrams = [[y  for y in x if y.lower() not in stops] for x in topNGrams]
	print topNGrams[0][:50]
	print topNGrams[1][:50]
	print topNGrams[2][:50]
	print 
	print len(topNGrams[0])
	print len(topNGrams[1])
	print len(topNGrams[2])
	topNGrams[0] = topNGrams[0][:3000]
	topNGrams[1] = topNGrams[1][:3000]
	topNGrams[2] = topNGrams[2][:3000]

	sw = stopwords.words('english')
	if 'the' in sw:
		print 'found'
	else:
		print 'not found' 

	return topNGrams

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

if __name__ == '__main__':
	main()

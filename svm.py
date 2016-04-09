import csv
from collections import defaultdict
import operator 
import cPickle
from itertools import islice
import sklearn.svm
from sklearn import svm
import numpy as np

TRAINING_INPUT_FILE = 'data/positive_negative_reviews_sentiment_2k.csv'
ngramDict = []

def main():
	global ngramDict
	# nltk.download()
	# prev = [''] * (3)
	# prev[0] = 'rohan'
	# print prev
	# return


	ngrams = getNGramsFromTrainingData()
	# print len(ngrams[0])
	# print len(ngrams[1])
	# print len(ngrams[2])

	with open(r"ngrams_v2.pickle", "wb") as output_file:
		cPickle.dump(ngrams, output_file)

	# with open(r"ngrams.pickle", "rb") as input_file:
	# 	ngrams = cPickle.load(input_file)

	ngrams[0] = ngrams[0][:10000]
	# ngrams[0] = ngrams[0][:1000]
	print len(ngrams[0])
	print len(ngrams[1])
	print len(ngrams[2])
	print ngrams[0][:10]
	print ngrams[1][:10]
	print ngrams[2][:10]
	return
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

	for i in range(10):

		start = i * (n/10)
		end = start + (n/10)
		accuracy = svc.fit(x[:start] + x[end:],y[:start] + y[end:]).score(x[start:end],y[start:end])
		# accuracy = svc.fit(x[1000:],y[1000:]).score(x[:1000],y[:1000])
		scores.append(accuracy)
		print accuracy
	
	print np.mean(np.array(scores))
	

	return


def populateNgramDict(nGrams):
	global ngramDict
	n = len(nGrams)
	ngramDict = [{} for i in range(n)]
	for i in range(n):
		for j,word in enumerate(nGrams[i]):
			ngramDict[i][word] = j


def buildFeatureTable():
	global TRAINING_INPUT_FILE
	global ngramDict
	length = [len(x) for x in ngramDict]
	featureVectors = []
	labels = []
	with open(TRAINING_INPUT_FILE, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i,row in enumerate(reader):
			if i == 0:
				continue
			if i%100 == 0:
				# print row[-1]
				print i
			instance = [0] * (sum(length))
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

				prev[0] = prev[1]
				prev[1] = word

			featureVectors.append(instance)
			labels.append(0 if row[-1] == 'negative' else 1)

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
			# print ', '.join(row)
			# print len(row)
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

				# for j in range(n):
				# 	print type(ngrams[j])
				# 	ngrams[j][word] = +1

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

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

if __name__ == '__main__':
	main()
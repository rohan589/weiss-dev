import csv_helper as csv
from collections import defaultdict
import operator 
import cPickle
from itertools import islice
import sklearn.svm
from sklearn import svm
import numpy as np
# from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.parse.stanford import StanfordParser # as stanford
from nltk.tokenize.stanford import StanfordTokenizer
# nltk.download()

def main():
	TRAINING_INPUT_FILE = 'data/positive_negative_reviews_sentiment_2k.csv'
	OUTPUT_FILE = 'data/positive_negative_trigrams_2k.csv'
	rows = csv.getRows(TRAINING_INPUT_FILE)
	cols = csv.getHeader(TRAINING_INPUT_FILE)
	cols.append('trigrams')
	for row in rows:
		row.append('dummy data')
	csv.writeFile(OUTPUT_FILE,rows,cols)
	print cols


	# parser = stanford.StanfordParser(model_path="/location/of/the/englishPCFG.ser.gz")
	parser = StanfordParser(model_path="/Users/rohankohli/Documents/workspace/CoreNLP/models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
	sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your name?"))
	print sentences
	print sentences.next()
	return

	EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
	print(sent_tokenize(EXAMPLE_TEXT))
	return

	# text = 'Punkt knows that the periods in Mr. Smith and Johann S. Bach do not mark sentence boundaries. And sometimes sentences can start with non-capitalized words.  i is a good variable name.'
	# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	# print('\n-----\n'.join(sent_detector.tokenize(text.strip())))



	return

if __name__ == '__main__':
	main()
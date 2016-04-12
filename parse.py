import csv_helper as csv
import pprint
import operator
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
from nltk.parse.stanford import StanfordParser
# from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
dependency_parser = None

def getSentenceTrigram(sentence):
	try:
		global dependency_parser
		dependency_parser = StanfordDependencyParser() if dependency_parser is None else dependency_parser
		# pp = pprint.PrettyPrinter(indent = 4)
		parse = getDependencyParseList(dependency_parser,sentence)
		# pp.pprint(parse)
		# print sentence
		if sentence[-1].lower() < 'a' or sentence[-1].lower() > 'z':
			sentence = sentence[:-1]
		print sentence
		sentenceString = sentence
		sentence = sentence.split()
		i = [-1,-1,-1]
		trigram = {}
		
		w1 = parse[0][0][0].encode('ascii','ignore')
		w2 = parse[0][2][0].encode('ascii','ignore')
		trigram[w1] = sentenceString.index(w1)
		trigram[w2] =  sentenceString.index(w2)
		# w1 = sentence[i[0]]
		for j in range(1,len(parse)):
			triple = parse[j]
			if (triple[0][0].encode('ascii','ignore') == w1):
				w3 = triple[2][0].encode('ascii','ignore')
				trigram[w3] = sentenceString.index(w3)
				break
		# words = [sentence[j] for j in sorted(i)]
		words = [x[0] for x in sorted(trigram.items(), key = operator.itemgetter(1))]
		print words
		return ' '.join(words)
	except:
		print 'execption caught. sentence = ' + str(sentence)
		return ''

def splitTextIntoSentences(dependency_parser, text):
	global sent_detector
	text = 'All of us went to the show to watch the entire band play live. Mangoes are liked by me.'
	sentences = sent_detector.tokenize(text)
	return sentences

def getSentenceTrigramsForText(text):
	global sent_detector
	global dependency_parser
	dependency_parser = StanfordDependencyParser() if dependency_parser is None else dependency_parser
	try:
		sentences = sent_detector.tokenize(text)
	except:
		print 'caught exception while tokenizing'
		return []
	
	print sentences
	# print sentences
	trigrams = [getSentenceTrigram(s) for s in sentences]
	# print sentences
	return trigrams

def printDependencyParse(dependency_parser,string):
	pp = pprint.PrettyPrinter(indent = 4)
	pp.pprint(getDependencyParseList(dependency_parser,string))


def getDependencyParseList(dependency_parser, string):
	result = (dependency_parser.raw_parse(string)).next()
	return list(result.triples())

def main():

	# getSentenceTrigramsForText(dependency_parser,'I shot an elephant in my sleep. All of us went to the show to watch the entire band play live.Mangoes are liked by me. The man in the black suit played good guitar. Either of these yields a good performance statistical parsing system. The movie was good to some extent but I did not like it.')
	# return

	tokenizer = PunktSentenceTokenizer()
	parser = StanfordParser()
	# sentences = tokenizer.tokenize(f.read().decode('utf-8').replace("\n"," "))
	# sentences = tokenizer.tokenize(row)
	# parseTree = list(parser.raw_parse((sentences[0])))

	TRAINING_INPUT_FILE = 'data/positive_negative_reviews_sentiment_2k.csv'
	OUTPUT_FILE = 'data/positive_negative_trigrams_2k.csv'

	header = csv.getHeader(TRAINING_INPUT_FILE)
	rows = csv.getRows(TRAINING_INPUT_FILE)
	cols = csv.getHeader(TRAINING_INPUT_FILE)
	cols.append('trigrams')
	print header
	for row in rows[:]:
		body = row[2]
		body = body.replace('\n',' s')
		trigrams = getSentenceTrigramsForText(body)
		print trigrams
		if trigrams is not None:
			trigrams = ','.join(trigrams)
		else:
			print 'trigrams is None'
			trigrams = ''
		
		row.append(trigrams)
		# for sentence in sentences:
			# parseTree = list(parser.raw_parse((sentence)))
			# trigrams.append()
			# print sentence
			# print parseTree
		
		

	csv.writeFile(OUTPUT_FILE,rows,cols)

	return


	
	

if __name__ == '__main__':
	main()

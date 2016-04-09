import csv_helper as csv
import pprint
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk
from nltk.parse.stanford import StanfordParser
# from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordDependencyParser


def getSetenceTrigram(dependency_parser,string):
	# string = string.encode('ascii','ignore')
	pp = pprint.PrettyPrinter(indent = 4)
	parse = getDependencyParseList(dependency_parser,string)
	string = string.split()
	# print parse[0][0]
	# print parse[0][2]
	i = [-1,-1,-1]
	i[0] = string.index(parse[0][0][0].encode('ascii','ignore'))
	i[1] = string.index(parse[0][2][0].encode('ascii','ignore'))
	w1 = string[i[0]]
	# print w1
	for j in range(1,len(parse)):
		triple = parse[j]
		if (triple[0][0].encode('ascii','ignore') == w1):
			i[2] = string.index(triple[2][0].encode('ascii','ignore'))
			break
	words = [string[j] for j in sorted(i)]
	print words
	return


def printDependencyParse(dependency_parser,string):
	pp = pprint.PrettyPrinter(indent = 4)
	pp.pprint(getDependencyParseList(dependency_parser,string))


def getDependencyParseList(dependency_parser, string):
	result = (dependency_parser.raw_parse_sents(string)).next()
	return list(result.triples())

def main():

	dependency_parser = StanfordDependencyParser()
	getSetenceTrigram(dependency_parser,'I shot an elephant in my sleep')
	print
	getSetenceTrigram(dependency_parser,'Mangoes are liked by me')
	print 
	getSetenceTrigram(dependency_parser,'All of us went to the show to watch the entire band play live')
	print 
	getSetenceTrigram(dependency_parser,'The man in the black suit played good guitar')
	print 
	getSetenceTrigram(dependency_parser,'Either of these yields a good performance statistical parsing system')
	print 
	getSetenceTrigram(dependency_parser,'The movie was good to some extent but I did not like it')
	return

	# dep_parser = StanfordNeuralDependencyParser()
	# print [parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")] 
	# [Tree('jumps', [Tree('fox', ['The', 'quick', 'brown']), Tree('dog', ['over', 'the', 'lazy'])])]
	return

	tokenizer = PunktSentenceTokenizer()
	parser = StanfordParser()
	# sentences = tokenizer.tokenize(f.read().decode('utf-8').replace("\n"," "))
	# sentences = tokenizer.tokenize(row)
	# parseTree = list(parser.raw_parse((sentences[0])))

	TRAINING_INPUT_FILE = 'data/positive_negative_reviews_sentiment_2k.csv'

	header = csv.getHeader(TRAINING_INPUT_FILE)
	rows = csv.getRows(TRAINING_INPUT_FILE)
	print header
	for row in rows[:5]:
		body = row[2]
		sentences = tokenizer.tokenize(body)
		for sentence in sentences:
			parseTree = list(parser.raw_parse((sentence)))
			print parseTree
		break
	return


	
	

if __name__ == '__main__':
	main()

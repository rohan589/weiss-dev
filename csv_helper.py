import csv
import numpy as np
# this is just like a helper module for reading CSV rows and column names

def main():
	a = [[1,2],[3,4],[5,6]]
	colHeader = [-1,-1]
	rowHeader = [0,0,0]
	writeFile('test.csv',a, colHeader,None)
	return


def writeFile(filename,array,colHeader = None, rowHeader = None):
	array = np.array(array)
	filename = filename # 'OUTPUT_' + 
	if colHeader is not None:
		colHeader = (np.array([colHeader]))
		array = np.vstack((colHeader,array))

	if rowHeader is not None:
		rowHeader = np.transpose(np.array([['']+rowHeader]))
		array = np.hstack((rowHeader,array))

	with open(filename, 'w',) as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(array)


def getHeader(INPUT_FILE):
	with open(INPUT_FILE, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i,row in enumerate(reader):
			return row

def getRows(INPUT_FILE):
	with open(INPUT_FILE, 'rU') as csvfile:
		table = []
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			break
		for i,row in enumerate(reader):
			table.append(row)
		return table


if __name__ == '__main__':
	main()

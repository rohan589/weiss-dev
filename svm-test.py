import sklearn.svm
from sklearn import svm
# from sklearn import datasets
# import numpy as np
# from scipy.sparse import *
# from scipy import *

def main():
	x = [[1,1,0,0,1,1], [1,1,0,0,1,0],[0,0,1,1,0,0],[0,1,1,1,0,0]]
	# print x.shape
	y = [1,1,0,0]
	print len(y)
	
	svc = svm.LinearSVC(C=1)
	print svc.fit(x[:],y[:]).score(x[:],y[:])

	return 

if __name__ == '__main__':
	main()
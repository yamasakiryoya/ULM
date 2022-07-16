#import
import numpy as np
import numpy.random as rd
import copy
rd.seed(1)


for K in range(3,11):
	print(K)
	prob=np.zeros((10**6,K))
	tmp=np.arange(K)
	#
	for i in range(10**6):
		prob[i,:] = rd.permutation(tmp)

	MU=0
	for i in range(10**6):
		flag = 1
		mp = np.argmax(prob[i,:])
		for k in range(mp):
			if prob[i,k] > prob[i,k+1]:
				flag = 0
		for k in range(mp, K-1):
			if prob[i,k] < prob[i,k+1]:
				flag = 0
		MU+=flag
	print(MU/10**6)


for K in range(3,11):
	print(K)
	prob=np.zeros((10**6,K))
	tmp=np.arange(K)
	#
	for i in range(10**6):
		prob[i,:] = rd.permutation(tmp)

	MU=0
	for i in range(10**6):
		flag = 1
		mp = np.argmax(prob[i,:])
		if mp>1:
			for k in range(mp-1):
				if prob[i,k]/prob[i,k+1] > prob[i,k+1]/prob[i,k+2]:
					flag = 0
		if mp<K-2:
			for k in range(mp+1, K-1):
				if prob[i,k]/prob[i,k-1] < prob[i,k+1]/prob[i,k]:
					flag = 0
		MU+=flag
	print(MU/10**6)

import theano
import theano.tensor as T 
import numpy
import random
from itertools import izip
import time
import sys


def mkBatch(xAll, yHatAll, dataSize, batchNumber):
	xBatch = []
	yHatBatch = []
	index = 0
	batchCnt = 0
	allData = len(yHatAll)

	batchSize = allData // batchNumber 
	if allData % batchNumber != 0:
		batchSize += 1

	flag = False
	while flag == False:
		if index >= allData:
			break
		xBatch.append([])
		yHatBatch.append([])

		# for yHatBatch
		for i in range(48):
			yHatBatch[batchCnt].append([])
			for j in range(index, index + batchSize):
				if j >= allData:
					flag = True
					break

				##### means that this should be 0
				if i != yHatAll[j]:
					yHatBatch[batchCnt][i].append(0)
				else:
					yHatBatch[batchCnt][i].append(1)

		# for xBatch
		for i in range(dataSize):
			xBatch[batchCnt].append([])
			for j in range(index, index + batchSize):
				if j >= allData:
					flag = True
					break
				xBatch[batchCnt][i].append(xAll[j][i])
			if flag:
				break
		index += batchSize
		batchCnt += 1
	x = []
	y_hat = []
	for i in xBatch:
		x.append(numpy.matrix(i,dtype='float32'))
	for i in yHatBatch:
		y_hat.append(numpy.matrix(i,dtype='float32'))

	return x, y_hat
		

def makeMapping(mapFile):
	mapping = {}
	for i in range(100000):
		tmpLine = mapFile.readline().strip()
		if tmpLine == "":
			break
		label = tmpLine.split()[0]
		mapping[label] = i
	return mapping


if __name__ == "__main__" :


	trainFile = open("mediumData", "r")
	labelFile = open("mediumLabel", "r")
	mapFile = open("48_39.map", "r")

	mapping = makeMapping(mapFile)

	xAll = []
	yHatAll = []
	while(True):
		tmpLine = trainFile.readline()
		if tmpLine == "":
			break
		features = tmpLine.split()
		features.pop(0)
		xAll.append(features)

		tmpLine = labelFile.readline().strip()
		label = tmpLine.split(",")
		yHatAll.append(mapping[label[1]])

	def MyUpdate(paramaters, momentum):
		#mu = numpy.float32(0.001)
		paramaters_update = \
		[(p, p + v) for p, v in izip(paramaters, momentum) ]
		return paramaters_update

	def VUpdate(momentum, gradients, learningRate):
		#mu = numpy.float32(learningRate)
		l = numpy.float32(0.9)
		paramaters_update = \
		[(v, l * v - learningRate * g) for v, g in izip(momentum, gradients) ]
		return paramaters_update
	def LRUpdate(learningRate):
		rateDecay = numpy.float32(0.9999)
		paramaters_update = \
		[(learningRate, rateDecay * learningRate)]
		return paramaters_update
	

	Vw1 = theano.shared(numpy.zeros((128,39), dtype='float32'))
	Vb1 = theano.shared(numpy.zeros((128),dtype='float32'))
	Vw2 = theano.shared(numpy.zeros((48,128), dtype='float32'))
	Vb2 = theano.shared(numpy.zeros((48),dtype='float32'))

	momentum = [Vw1, Vb1, Vw2, Vb2]

	x = T.matrix(dtype='float32')
	w1 = theano.shared(numpy.random.normal(0,0.1,(128,39)).astype(dtype='float32'))
	b1 = theano.shared(numpy.random.normal(0,0.1,128).astype(dtype='float32'))
	b2 = theano.shared(numpy.random.normal(0,0.1,48).astype(dtype='float32'))
	w2 = theano.shared(numpy.random.normal(0,0.1,(48,128)).astype(dtype='float32'))
	
	z1 = T.dot(w1,x) + b1.dimshuffle(0,'x')
	a1 = 1/(1+T.exp(-z1))
	z2 = T.dot(w2, a1) + b2.dimshuffle(0,'x')
	y = 1/(1+T.exp(-z2))
	
	y_hat = T.matrix(dtype='float32')
	cost = T.sum((y-y_hat)**2)

	gradients = T.grad(cost, [w1, b1, w2, b2])

	neuron1 = theano.function(inputs=[x],outputs=a1)
	neuron2 = theano.function(inputs=[a1],outputs=y)

	#mu = numpy.float32(0.1)
	#print mu
	#print type(mu)
	de = theano.shared(numpy.float32(0.0001))
	Decay = theano.function(inputs=[],updates=LRUpdate(de))
	movement = theano.function(inputs=[x, y_hat], updates=VUpdate(momentum, gradients, de))
	train = theano.function(inputs=[x, y_hat], updates=MyUpdate([w1, b1, w2, b2], momentum), outputs=cost)
	test = theano.function(inputs=[x], outputs=y)
	validation = theano.function(inputs=[x, y_hat], outputs=cost)
	#x = numpy.matrix([[1,1],[-1,1],[1,1]], dtype='float32')#[[1, -1, 1],[1, 1, 1]]
	#x = numpy.array(x).astype(dtype='float32')
	#print x.type.dtype
	#y_hat = numpy.matrix([[0,1],[1,0],[0,0]], dtype='float32')#[[0, 1, 0],[1, 0, 0]]
	#x = numpy.array(y_hat).astype(dtype='float32')
	valiMini = 1000000000000000
	minCost = 100000000000000
	dataSize = len(xAll[0])
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, 100)
	for t in range(50000):
		cost = 0
		valiCost = 0
		for i in range(100):
			movement(xBatch[i],yHatBatch[i])
			cost += train(xBatch[i],yHatBatch[i])
			Decay()
		cost/=100
		if cost<minCost:
			minCost = cost
		elif cost>=minCost and t>100:
			break


		# if cost < 90:
		# 	break
		print >> sys.stderr, cost

	for i in range(100):
		print xBatch[i]
		print yHatBatch[i]
		print "-------------------"
	
	# s = time.time()
	# for i in range(1000):
	# 	print "--------"+str(i+1)+"--------"
	# 	train(x, y_hat)
	# 	test(x)
	# 	print w.get_value(), b.get_value()
	# 	print wy.get_value()
	# f = time.time()

	# print f-s
	
	
	"""
	gradient = theano.function(inputs=[x, y_hat], outputs=[dw, db])

	x = [1, -1]
	y_hat = 1
	for i in range(10000):
		print neuron(x)
		dw, db = gradient(x, y_hat)
		w.set_value(w.get_value() - 0.1 * dw)
		b.set_value(b.get_value() - 0.1 * db)
		print w.get_value(), b.get_value()
	"""
 
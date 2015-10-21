import theano
import theano.tensor as T 
import numpy
import random
from itertools import izip
import time
import sys


def valid(yBatch, yHatBatch, batchSize):
	yHat = []
	y = []

	for i in range(batchSize):
		maxValue = 0
		maxIndex = 0

		for j in range(48):
			if yBatch[j][i] > maxValue:
				maxValue = yBatch[j][i]
				maxIndex = j
			if yHatBatch.item((j,i)) == 1:
				yHat.append(j)		
		y.append(maxIndex)
	

	print "yHat: " 
	print yHat
	print "y:    " 
	print y	

	err = 0
	for i in range(len(yHat)):
		if yHat[i] != y[i]:
			err += 1

	return err

def remap(yBatch, yHatBatch, batchSize):
	yHat = []
	y = []

	for i in range(batchSize):
		maxValue = 0
		maxIndex = 0

		for j in range(48):
			if yBatch[j][i] > maxValue:
				maxValue = yBatch[j][i]
				maxIndex = j		
		y.append(maxIndex)

	return y

def mkBatch(xAll, yHatAll, dataSize, batchNumber):
	xBatch = []
	yHatBatch = []
	index = 0
	batchCnt = 0
	allData = len(xAll)

	batchSize = allData // batchNumber 
	if allData % batchNumber != 0:
		batchSize += 1

	flag = False
	while flag == False:
		if index >= allData:
			break
		xBatch.append([])
		yHatBatch.append([])

		
		for i in range(index, index + batchSize):
			if i >= allData:
				flag = True
				break
			# for yHatBatch
			tmp = numpy.zeros(48)
			tmp[yHatAll[i]] = 1
			yHatBatch[batchCnt].append(tmp)

			# for xBatch
			xBatch[batchCnt].append(xAll[i])
		index += batchSize
		batchCnt += 1
	x = []
	y_hat = []
	for i in xBatch:
		transpose = numpy.transpose(numpy.matrix(i,dtype='float32'))
		x.append(transpose)
	for i in yHatBatch:
		transpose = numpy.transpose(numpy.matrix(i,dtype='float32'))
		y_hat.append(transpose)

	return x, y_hat
		

def makeMapping(mapFile):
	mapping = {}
	remapping = {}
	for i in range(100000):
		tmpLine = mapFile.readline().strip()
		if tmpLine == "":
			break
		label = tmpLine.split()
		mapping[label[0]] = i
		remapping[i] = label[1]
	return mapping, remapping

def MyUpdate(paramaters, gradients):
		mu = numpy.float32(0.1)
		paramaters_update = \
		[(p, p - mu * g) for p, g in izip(paramaters, gradients) ]
		return paramaters_update
def MMyUpdate(paramaters, momentum):
	#mu = numpy.float32(0.001)
	paramaters_update = \
	[(p, p + v) for p, v in izip(paramaters, momentum) ]
	return paramaters_update

def MVUpdate(momentum, gradients, learningRate):
	#mu = numpy.float32(learningRate)
	l = numpy.float32(0.9)
	paramaters_update = \
	[(v, l * v - learningRate * g) for v, g in izip(momentum, gradients) ]
	return paramaters_update
def MLRUpdate(learningRate):
	rateDecay = numpy.float32(0.9999)
	paramaters_update = \
	[(learningRate, rateDecay * learningRate)]
	return paramaters_update


if __name__ == "__main__" :


	trainFile = open("Validation_data_50000", "r")
	labelFile = open("Validation_label_50000", "r")
	mapFile = open("48_39.map", "r")

	mapping, remapping = makeMapping(mapFile)

	xAll = []
	yHatAll = []
	while(True):
		tmpLine = trainFile.readline().strip()
		if tmpLine == "":
			break
		features = tmpLine.split()
		features.pop(0)
		xAll.append(features)

		tmpLine = labelFile.readline().strip()
		label = tmpLine.split()
		yHatAll.append(mapping[label[1]])

	Vw1 = theano.shared(numpy.zeros((128,69), dtype='float32'))
	Vb1 = theano.shared(numpy.zeros((128),dtype='float32'))
	Vw2 = theano.shared(numpy.zeros((48,128), dtype='float32'))
	Vb2 = theano.shared(numpy.zeros((48),dtype='float32'))
	momentum = [Vw1, Vb1, Vw2, Vb2]
	
	# x = T.matrix(dtype='float32')
	# w = theano.shared(numpy.random.randn(128,39).astype(dtype='float32'))
	# b1 = theano.shared(numpy.ones((128),dtype='float32'))
	# b2 = theano.shared(numpy.ones((48),dtype='float32'))
	# wy = theano.shared(numpy.random.randn(48,128).astype(dtype='float32'))
	x = T.matrix(dtype='float32')
	w1 = theano.shared(numpy.random.normal(0,0.1,(128,69)).astype(dtype='float32'))
	b1 = theano.shared(numpy.random.normal(0,0.1,128).astype(dtype='float32'))
	b2 = theano.shared(numpy.random.normal(0,0.1,48).astype(dtype='float32'))
	w2 = theano.shared(numpy.random.normal(0,0.1,(48,128)).astype(dtype='float32'))
	
	z1 = T.dot(w1,x) + b1.dimshuffle(0,'x')
	a1 = 1/(1+T.exp(-z1))
	#a1 = (T.exp(2*z1)-1)/(T.exp(2*z1)+1)
	z2 = T.dot(w2, a1) + b2.dimshuffle(0,'x')
	y = T.exp(-z2)/T.sum(T.exp(-z2))
	#y = 1/(1+T.exp(-z2))
	
	y_hat = T.matrix(dtype='float32')
	#a = y_hat.nonzero()[1]
	#cost = -T.mean(T.log(y)[a,T.arange(y_hat.shape[1])])
	cost = -T.mean(T.log(y)*y_hat)
	#cost = T.sum((y-y_hat)**2)

	gradients = T.grad(cost, [w1, b1, w2, b2])

	neuron1 = theano.function(inputs=[x],outputs=a1)
	neuron2 = theano.function(inputs=[a1],outputs=y)

	#mu = numpy.float32(0.1)
	#print mu
	#print type(mu)
	train = theano.function(inputs=[x, y_hat], updates=MyUpdate([w1, b1, w2, b2], gradients), outputs=cost)
	test = theano.function(inputs=[x], outputs=y)
	#ValiCost = theano.function(inputs=[x, y_hat], outputs=cost)


	de = theano.shared(numpy.float32(0.0001))
	MDecay = theano.function(inputs=[],updates=MLRUpdate(de))
	Mmovement = theano.function(inputs=[x, y_hat], updates=MVUpdate(momentum, gradients, de))
	Mtrain = theano.function(inputs=[x, y_hat], updates=MMyUpdate([w1, b1, w2, b2], momentum), outputs=cost)
	#x = numpy.matrix([[1,1],[-1,1],[1,1]], dtype='float32')#[[1, -1, 1],[1, 1, 1]]
	#x = numpy.array(x).astype(dtype='float32')
	#print x.type.dtype
	#y_hat = numpy.matrix([[0,1],[1,0],[0,0]], dtype='float32')#[[0, 1, 0],[1, 0, 0]]
	#x = numpy.array(y_hat).astype(dtype='float32')
	batchNumber = 500#3581#12929
	# ValibatchNumber = 
	# validDataSize = 
	# MiniCost = 1000000
	s = time.time()
	dataSize = len(xAll[0])
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, batchNumber)
	print >> sys.stderr, "Time: "+str(time.time()-s)
	print >> sys.stderr, "done loading data"

	for t in range(10):
		cost = 0
		s = time.time()
		for i in range(batchNumber):
			# MDecay()
			# Mmovement(xBatch[i],yHatBatch[i])
			# cost += Mtrain(xBatch[i],yHatBatch[i])
			cost += train(xBatch[i],yHatBatch[i])
		cost /= batchNumber
		print >> sys.stderr, "iteration: "+str(t)
		print >> sys.stderr, "Time: "+str(time.time()-s)
		print >> sys.stderr, cost

	print >> sys.stderr, "done training"

	#trainFile = open("./MLDS/fbank/test.ark", "r")
	#labelFile = open("./Data/validation_label_50000", "r")
	mapFile = open("48_39.map", "r")

	mapping, remapping = makeMapping(mapFile)

	error = 0
	for i in range(batchNumber):
		error += valid(test(xBatch[i]), yHatBatch[i], 100)

	print >> sys.stderr, "error num: "+str(error)


	trainFile = open("validation_data_50000", "r")
	labelFile = open("validation_label_50000", "r")

	xAll = []
	yHatAll = []
	while(True):
		tmpLine = trainFile.readline().strip()
		if tmpLine == "":
			break
		features = tmpLine.split()
		features.pop(0)
		xAll.append(features)

		tmpLine = labelFile.readline().strip()
		label = tmpLine.split()
		yHatAll.append(mapping[label[1]])

	dataSize = len(xAll[0])
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, 1)

	error = valid(test(xBatch[0]), yHatBatch[0], 50000)

	print >> sys.stderr, "error num: "+str(error)







		# Vacost = 0
		# Error = 0
		# for i in range(ValibatchNumber):
		# 	Error += valid(test(xBatch[i]), yHatBatch[i], 6)
		# print >> sys.stderr, Error/validDataSize

		# for i in range(ValibatchNumber):


		# print "--------------------------------"
	
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
 
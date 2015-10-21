import theano
import theano.tensor as T 
import numpy
import random
from itertools import izip
import time
import sys
from theano.compile.debugmode import DebugMode


def valid(yBatch, yHatBatch, batchSize):
	yHat = []
	y = []

	TPyBatch = numpy.transpose(yBatch)
	TPyHatBatch = numpy.transpose(yHatBatch)

	for i in range(batchSize):
		yHat.append(numpy.argmax(TPyHatBatch[i]))
		y.append(numpy.argmax(TPyBatch[i]))

	print "yHat: " 
	print yHat
	print "y:    " 
	print y	

	err = 0
	for i in range(len(yHat)):
		if yHat[i] != y[i]:
			err += 1

	return err

def remap(yBatch, batchSize):
	y = []
	TPyBatch = numpy.transpose(yBatch)
	for i in range(batchSize):
		y.append(numpy.argmax(TPyBatch[i]))
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
			if len(yHatAll) > 0:
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

	# loading training and testing data 

	batchNumber = 0
	batchSize = 0

	TRAINFILE = "./Data/Train.data"
	LABELFILE = "./Data/Train.label"
	if TRAINFILE == "./Data/Train.data":
		batchNumber = 38787
		batchSize = 29
	elif TRAINFILE == "./Data/ValidationData":
		batchNumber = 3581
		batchSize = 63


	trainFile = open(TRAINFILE, "r")
	labelFile = open(LABELFILE, "r")

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

	# init movement variable

	Vw1 = theano.shared(numpy.zeros((256,69), dtype='float32'))
	Vb1 = theano.shared(numpy.zeros((256),dtype='float32'))
	Vw2 = theano.shared(numpy.zeros((48,256), dtype='float32'))
	Vb2 = theano.shared(numpy.zeros((48),dtype='float32'))
	momentum = [Vw1, Vb1, Vw2, Vb2]
	
	# init dnn variable

	x = T.matrix(dtype='float32')
	b1 = theano.shared(numpy.random.normal(0,0.1,256).astype(dtype='float32'))
	w1 = theano.shared(numpy.random.normal(0,0.1,(256,69)).astype(dtype='float32'))
	b2 = theano.shared(numpy.random.normal(0,0.1,256).astype(dtype='float32'))
	w2 = theano.shared(numpy.random.normal(0,0.1,(256,256)).astype(dtype='float32'))
	b3 = theano.shared(numpy.random.normal(0,0.1,48).astype(dtype='float32'))
	w3 = theano.shared(numpy.random.normal(0,0.1,(48,256)).astype(dtype='float32'))
	
	z1 = T.dot(w1,x) + b1.dimshuffle(0,'x')
	a1 = T.switch(z1<0,0,z1)#1/(1+T.exp(-z1))
	#a1 = T.switch(z1<0.0001,0.0001,z1)#(T.exp(2*z1)-1)/(T.exp(2*z1)+1)
	#a1 = T.switch(z1>0.9999,0.9999,z1)

	z2 = T.dot(w2,a1) + b2.dimshuffle(0,'x')
	a2 = T.switch(z2<0,0,z2)#1/(1+T.exp(-z2))#T.switch(z2<0.0001,0.0001,z2)#(T.exp(2*z2)-1)/(T.exp(2*z2)+1)
	#a2 = T.switch(z2>0.9999,0.9999,z2)
	#a1 = (T.exp(2*z1)-1)/(T.exp(2*z1)+1) # tan
	z3 = T.dot(w3,a2) + b3.dimshuffle(0,'x')
	y = T.exp(-z3)/T.sum(T.exp(-z3))
	#1/(1+T.exp(-z3))#T.exp(-z3)/T.sum(T.exp(-z3))
	
	
	y_hat = T.matrix(dtype='float32')
	cost = -T.mean(T.log(y)*y_hat) # softmax loss function
	#cost = T.sum((y-y_hat)**2)

	gradients = T.grad(cost, [w1, b1, w2, b2, w3, b3])

	# neuron1 = theano.function(inputs=[x],outputs=a1)
	# neuron2 = theano.function(inputs=[a1],outputs=a2)
	# neuron3 = theano.function(inputs=[a2],outputs=y)
	# IT = T.scalar()

	# itera = numpy.float32(0.1*math.sqrt(IT))

	train = theano.function(inputs=[x, y_hat], updates=MyUpdate([w1, b1, w2, b2, w3, b3], gradients), outputs=cost)
	test = theano.function(inputs=[x], outputs=y)



	#batchNumber = 12929
	s = time.time()
	dataSize = len(xAll[0])
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, batchNumber)
	print >> sys.stderr, "Time: "+str(time.time()-s)
	print >> sys.stderr, "done loading data"

	for t in range(100):
		cost = 0
		s = time.time()
		for i in range(batchNumber):
			cost += train(xBatch[i],yHatBatch[i])
		cost /= batchNumber
		print >> sys.stderr, "iteration: "+str(t)
		print >> sys.stderr, "Time: "+str(time.time()-s)
		print >> sys.stderr, cost

	print >> sys.stderr, "done training"

	error = 0
	for i in range(batchNumber):
		error += valid(test(xBatch[i]), yHatBatch[i], batchSize)
		
	errorRate = error/float(batchNumber*batchSize)
	print >> sys.stderr, "Ein: "+str(errorRate) 
	
	print >> sys.stderr, "start loading test data"

	trainFile = open("./MLDS/fbank/test.ark", "r")

	xAll = []
	yHatAll = []
	while(True):
		tmpLine = trainFile.readline().strip()
		if tmpLine == "":
			break
		features = tmpLine.split()
		features.pop(0)
		xAll.append(features)

	dataSize = len(xAll[0])
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, 1)

	print >> sys.stderr, "done loading test data"

	f = open("test_ans","w")

	ans = remap(test(xBatch[0]), 180406)

	for i in ans:
		f.write(remapping[i]+"\n")

	print >> sys.stderr, "done writting test to outputs"

	print >> sys.stderr, "start loading validation data"


	trainFile = open("./Data/validation_data_50000", "r")
	labelFile = open("./Data/validation_label_50000", "r")

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

	print >> sys.stderr, "Eout: "+str(error/50000.0)

	print >> sys.stderr, "done"








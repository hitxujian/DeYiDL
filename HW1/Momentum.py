import theano
import theano.tensor as T 
import numpy as np
import random
from itertools import izip
import time
import sys
import math

def makeMapping(MapFile):
	mapping = {}
	remapping = {}
	with open(MapFile, "r") as f:
		for i, line in enumerate(f):
			label = line.strip().split()
			mapping[label[0]] = i
			remapping[i] = label[1]
	return mapping, remapping
	
def loadData(TrainFile, LabelFile, mapping):
	xAll = []
	yHatAll = []

	#load X
	with open(TrainFile, "r") as f:
		for line in f:
			features = line.strip().split()
			#pop the data name
			features.pop(0)
			xAll.append([np.float32(i) for i in features])

	#load Label
	with open(LabelFile, "r") as f:
		for line in f:
			label = line.strip().split()
			#label = [name, label]
			yHatAll.append(mapping[label[1]])

	return xAll, yHatAll

def loadTestData(TestFile):
	xAll = []

	#load X
	with open(TestFile, "r") as f:
		for line in f:
			features = line.strip().split()
			#pop the data name
			features.pop(0)
			xAll.append([np.float32(i) for i in features])

	return xAll

def MyUpdate(paramaters, gradients):
	
	#mu = learning rate
	mu = np.float32(0.1)

	paramaters_update = \
	[(p, p - mu * g) for p, g in izip(paramaters, gradients) ] 
	return paramaters_update
#print >> sys.stderr, "learning rate: ", mu.get_value()
def MyUpdateMu(paramaters, momentum):
	
	#mu = learning rate

	paramaters_update = \
	[(p, p + v) for p, v in izip(paramaters, momentum) ] 
	return paramaters_update

def buildDNN(Neuron_Distribution):

	x = T.matrix(dtype='float32')
	y_hat = T.matrix(dtype='float32')
	WB_Parameters = []
	V_Parameters = []
	Neurons = []
	input_now = x
	for i in xrange(len(Neuron_Distribution)-1):
		dim_now = Neuron_Distribution[i]
		dim_next = Neuron_Distribution[i+1]
		
		#use gaussian(0, 0,1)
		#w = theano.shared(np.random.normal(0, 0.1, (dim_next, dim_now)).astype(dtype='float32'))
		#b = theano.shared(np.random.normal(0, 0.1, dim_next).astype(dtype='float32'))
		#use magic initial!!
		#+-sqrt(6/(#in+#out))
		interval = math.sqrt(6./float(dim_next+dim_now))/3.
		print interval
		#w = theano.shared(np.random.uniform(-interval, interval, (dim_next, dim_now)).astype(dtype='float32'))
		#b = theano.shared(np.random.uniform(-interval, interval, dim_next).astype(dtype='float32'))
		w = theano.shared(np.random.normal(0, interval, (dim_next, dim_now)).astype(dtype='float32'))
		b = theano.shared(np.random.normal(0, interval, dim_next).astype(dtype='float32'))
		
		WB_Parameters.append(w)
		WB_Parameters.append(b)

		Vw = theano.shared(np.zeros((dim_next,dim_now), dtype='float32'))
		Vb = theano.shared(np.zeros((dim_next), dtype='float32'))

		V_Parameters.append(Vw)
		V_Parameters.append(Vb)

		z = T.dot(w, input_now) + b.dimshuffle(0, 'x')

		#use logistic function
		#a = 1/(1 + T.exp(-z))
		#use switch!!
		a = T.switch(z<0,0,z)

		if i != len(Neuron_Distribution)-2:
			Neurons.append(theano.function(inputs=[input_now], outputs=a))
			input_now = a

	#y=softmax layer
	y = T.exp(-z) / T.sum(T.exp(-z))
	Neurons.append(theano.function(inputs=[input_now], outputs=y))

	#cost = perplexity
	cost = -T.mean(T.log(y) * y_hat)

	gradients = T.grad(cost, WB_Parameters)


	#mu = learning rate, for MyUpdateMu
	mu_initial = theano.shared(np.float32(0.2))
	t = T.scalar(dtype='float32')
	mu_decay = theano.function(inputs=[t], outputs=mu_initial/T.sqrt(t+1))
	mu = mu_initial/T.sqrt(t+np.float32(1.))

	Lambda = np.float32(0.7)
	#momentum = [v = Lambda * v + mu * g for v, g in izip(V_Parameters, gradients)]
	Momentum = theano.function(inputs=[x, y_hat, t], updates=[(v, Lambda * v - mu * g) for v, g in izip(V_Parameters, gradients)])


	#train = theano.function(inputs=[x, y_hat], updates=MyUpdate(WB_Parameters, gradients), outputs=cost)
	train = theano.function(inputs=[x, y_hat], updates=MyUpdateMu(WB_Parameters, V_Parameters), outputs=cost)
	test = theano.function(inputs=[x], outputs=y)

	return train, test, mu_decay, Momentum

def mkBatch(xAll, yHatAll, batchNumber, Neuron_Distribution):
	x = []
	y_hat = []

	#ansDim = feature number of an y 
	ansDim = Neuron_Distribution[-1]
	#dataSize = number of x
	dataSize = len(xAll)
	#batchSize = length of a batch
	batchSize = int(math.ceil(dataSize / batchNumber))
	for i in xrange(batchNumber):

		#for y_hat batch
		yHatBatch = yHatAll[i*batchSize:(i+1)*batchSize]

		yHatBatch_matrix = np.matrix(np.zeros((ansDim, len(yHatBatch)), dtype='float32'))
		for index, yHat in enumerate(yHatBatch):
			yHatBatch_matrix[yHat, index] = 1.

		y_hat.append(yHatBatch_matrix)

		#for x batch
		xBatch = xAll[i*batchSize:(i+1)*batchSize]
		xBatch_matrix = np.transpose(np.matrix(xBatch, dtype='float32'))
		x.append(xBatch_matrix)

	return x, y_hat

def valid(yBatch, yHatBatch):

	#here yBatch is a batch, not a list of batches
	y = np.argmax(yBatch, axis=0)
	yHat = np.argmax(yHatBatch, axis=0)
	
	error = 0.
	for i in xrange(len(y)):
		#because y is array, but yHat is matrix
		if y[i] != yHat[0, i]:
			#0/1 error type
			error += 1.

	return error

def remap(yTest):
	
	y = np.argmax(yTest, axis=0)

	return y

if __name__ == "__main__" :

	#-----------------------------------------------------------------#
	#-----------------------loading data------------------------------#

	TrainFile = "./Data/Train.data"
	LabelFile = "./Data/Train.label"
	MapFile = "./48_39.map"

	#mapping = label1 to index, remapping = index to label2, 
	#ex: mapping:{epi:16}, remapping:{16:sil} for (epi, sil) pair
	mapping, remapping = makeMapping(MapFile)

	#load X and Label(mapped to index)
	xAll, yHatAll = loadData(TrainFile, LabelFile, mapping)
	
	Neuron_Distribution = [69, 256, 256, 256, 48]
	#say there is a 69 * 128 * 48 DNN
	#the first item is input dimension, the final item is output dimension
	#to add hidden layer, just add in this list, e.g. [69, 128, 128, 48]
	
	#now build a gaussian(0,0.1) and use logistic function, cost function = perplexity
	train, test, mu_decay, Momentum = buildDNN(Neuron_Distribution)
	
	batchNumber = 38787

	s = time.time()
	#we need Neuron_Distribution here because we can not get the dimension of
	#yHatBatch by yHatBatch itself
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, batchNumber, Neuron_Distribution)
	
	print >> sys.stderr, "time: "+str(time.time()-s)
	print >> sys.stderr, "done loading data"

	#-----------------------------------------------------------------#
	#-----------------------training start----------------------------#
	
	Iterations = 100

	for t in xrange(Iterations):
		cost = 0.
		s = time.time()

		for i in xrange(batchNumber):
			Momentum(xBatch[i], yHatBatch[i], t)
			cost += train(xBatch[i], yHatBatch[i])

		cost /= batchNumber
		print >> sys.stderr, "iteration: "+str(t)
		print >> sys.stderr, "mu: "+str(mu_decay(t))
		print >> sys.stderr, "time: "+str(time.time()-s)
		print >> sys.stderr, cost

	print >> sys.stderr, "done training"

	#Ein
	error = 0.
	for i in range(batchNumber):
		error += valid(test(xBatch[i]), yHatBatch[i])

	print >> sys.stderr, "error num: "+str(error)
	errorRate = error/float(len(yHatAll))
	print >> sys.stderr, "Ein: "+str(errorRate) 

	#-----------------------------------------------------------------#
	#-----------------------testing start-----------------------------#

	TestFile = "./MLDS/fbank/test.ark"
	MapFile = "./48_39.map"
	AnsFile = "Mtest.ans"

	mapping, remapping = makeMapping(MapFile)



	xAll = loadTestData(TestFile)

	#here use the whole xAll for test
	xTest = np.transpose(np.matrix(xAll))
	

	print >> sys.stderr, "done loading test data"

	ans = remap(test(xTest))

	with open(AnsFile, "w") as p:
		for i in ans:
			p.write(remapping[i]+'\n')

	print >> sys.stderr, "done writting answer"

	#-----------------------------------------------------------------#
	#-----------------------Validation_50000--------------------------#

	TrainFile = "./Data/validation_data_50000"
	LabelFile = "./Data/validation_label_50000"
	MapFile = "./48_39.map"

	mapping, remapping = makeMapping(MapFile)

	xAll, yHatAll = loadData(TrainFile, LabelFile, mapping)

	batchNumber = 1
	xBatch, yHatBatch = mkBatch(xAll, yHatAll, batchNumber, Neuron_Distribution)

	error += valid(test(xBatch[0]), yHatBatch[0])

	print >> sys.stderr, "error num: "+str(error)













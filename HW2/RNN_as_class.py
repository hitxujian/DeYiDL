import theano
import theano.tensor as T 
import numpy as np
import random
from itertools import izip
import time
import sys
import cPickle as pickle

class RNN(object):
	# @params contains: [Neuron distribution, initial learning rate,
	# 					activation function, cost function, epochs]
	def __init__(self, params):
		self.WB_parameters = []
		self.INIT_parameters = []

		# learning rate
		self.mu = float(params[1])
		if params[2] == "sigmoid":
			self.activation = sigmoid
		elif params[2] == "tanh":
			self.activation = tanh
		else:
			self.activation = ReLU

		self.costFunc = params[3]
		
		self.epochs = int(params[4])

		self.Neuron_Distribution = params[0]
		for i in xrange(len(self.Neuron_Distribution)-1): # i = 0, 1, 2
			dim_now = self.Neuron_Distribution[i]
			dim_next = self.Neuron_Distribution[i+1]

			#w,b means main layer
			tmpW = theano.shared(np.random.normal(0, 0.1, (dim_now, dim_next)))
			tmpB = theano.shared(np.random.normal(0, 0.1, (dim_next)))
			self.WB_parameters += [tmpW, tmpB]

			#h means output, in this case, h0, h1, y
			h = theano.shared(np.zeros(dim_next))
			self.INIT_parameters.append(h)

			
			if i != len(Neuron_Distribution)-2:
				#wh, bh means hidden memory layer
				#wh = theano.shared( np.random.normal(0,1, (dim_next, dim_next)))
				tmpWh = theano.shared(np.ones((dim_next, dim_next)))
				tmpBh = theano.shared(np.zeros(dim_next))
				self.WB_parameters += [tmpWh, tmpBh]

		
		

	def loadModel(self, fpath):
		f = file(fpath, 'rb')
		self.WB_parameters = pickle.load(f)
		f.close()

	def saveModel(self, fpath):
		f = file(fpath, 'wb')
		pickle.dump(self.WB_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
		f.close()

	def buildRNN(self):
		x_seq = T.matrix('input')
		y_hat = T.matrix('target')
			
		#to pass the all parameters to scan function, pack those parameter (encoding)
		OUTPUTS_INFO, NON_SEQ = self.ScanParaPacker(self.INIT_parameters, self.WB_parameters)
		
		HY_seq,_ = theano.scan(
							self.step, \
							sequences = x_seq, \
							outputs_info =  OUTPUTS_INFO, \
							non_sequences = NON_SEQ, \
							truncate_gradient=-1 \
		)

		y_seq = HY_seq[-1] #HY_seq[-1] = y_seq

		#y_seq_last = y_seq[-1][0] # we only care about the last output, data dependent
		y_seq_last = y_seq

		if self.costFunc == "cross entropy":
			self.costFunc = -T.mean(T.log(y_seq_last) * y_hat)
		elif self.costFunc == "Euclidean distance":
			self.costFunc = T.sum( ( y_seq_last - y_hat )**2 ) 
		# As default
		else:
			self.costFunc = T.sum( ( y_seq_last - y_hat )**2 ) 


		"""
		Cost Function decision based on params
		"""
		
		gradients = T.grad(self.costFunc, self.WB_parameters)

		test = theano.function(
			inputs= [x_seq], \
			outputs=y_seq_last \
		)

		train = theano.function(
				inputs=[x_seq,y_hat], \
				outputs=self.costFunc, \
				updates=self.MyUpdate(self.WB_parameters,gradients) \
		)

		self.testFunc = test
		self.trainFunc = train

		print >> sys.stderr, "Done building RNN"


	def train(self, data, labels):
		Iterations = 5

		for t in range(self.epochs):
			
			cost = 0.
			s = time.time()

			for i in xrange(len(data)):
				cost += self.trainFunc(data[i], labels[i])

			print >> sys.stderr, "iteration:", t
			print >> sys.stderr, "time:", time.time() - s
			print >> sys.stderr, "cost:", cost

		print >> sys.stderr, time.time(), "Done training"

		#count Ein...
		error, errorRate = self.test(data, labels)
		print >> sys.stderr, "error count:", error
		print >> sys.stderr, "Ein: " + str(errorRate) 
		print >> sys.stderr, time.time()

	"""
	.test() take data and labels to determine the errors and error rate
	"""
	def test(self, data, labels):
		error = 0.
		num = 0.
		for i in xrange(len(data)):
			error += self.valid(self.testFunc(data[i]), labels[i])
			num += labels[i].shape[0]
		errorRate = error / num
		return error, errorRate
		

	def step(self, x_t, *arg):

		#x_t,INIT_parameters, WB_parameters
		
		#decoding, in init_para: [h0,h1,...,y], in wb_para: [w0,b0,wh0,bh0,w1,b1,wh1,bh1,...,wo,bo]
		INIT_parameters, WB_parameters = self.ScanParaParser(arg)
		
		HY_parameters = []
		now_input = x_t
		for i in xrange(len(INIT_parameters)-1):
			[wi, bi, wh, bh] = WB_parameters[4*i:4*i+4]
			h_tm1 = INIT_parameters[i]
			z = T.dot(now_input,wi) + bi \
					+ T.dot(h_tm1, wh) + bh

			

			h_t = self.activation(z)
			#h_t = T.switch(z<0, 0, z) #ReLU

			HY_parameters.append(h_t)
			now_input = h_t
		
		wo = WB_parameters[-2]
		bo = WB_parameters[-1]

		y_t = softmax(T.dot(h_t, wo) + bo)
		HY_parameters.append(y_t)
		
		return HY_parameters


	def MyUpdate(self, parameters, gradients):
		parameters_updates = [(p, p - self.mu * g) for p,g in izip(parameters, gradients)] 
		return parameters_updates


	def valid(self, ySeq, yHatSeq):

		#here yBatch is a batch, not a list of batches
		y = np.argmax(ySeq, axis=1)
		yHat = np.argmax(yHatSeq, axis=1)

		error = 0.
		for i in xrange(len(y)):
			#because y is array, but yHat is matrix
			if y[i] != yHat[i, 0]:
				#0/1 error type
				error += 1.

		return error

	def remap(self, yTest):
		
		y = np.argmax(yTest, axis=1)

		return y



	def ScanParaPacker(self, INIT_parameters, WB_parameters):

		return INIT_parameters, [len(WB_parameters)] + WB_parameters + [len(INIT_parameters)]
		
	def ScanParaParser(self, arg):
		
		init_len = int(T.get_scalar_constant_value(arg[-1]))
		wb_len = int(T.get_scalar_constant_value(arg[0+init_len]))

		INIT_parameters= [arg[i] for i in xrange(0,0+init_len)]
		WB_parameters = [arg[i] for i in xrange(0+init_len+1,0+init_len+1+wb_len)]

		return INIT_parameters, WB_parameters
##########################################################


def makeMapping(MapFile):
	mapping = {}
	remapping = {}
	with open(MapFile, "r") as f:
		for i, line in enumerate(f):
			label = line.strip().split()
			mapping[label[0]] = i
			remapping[i] = label[1]
	return mapping, remapping

def sigmoid(z):
	return 1/(1+T.exp(-z))

def softmax(z):
	return T.exp(-z) / T.sum(T.exp(-z))

def ReLU(z):
	return T.switch(z<0, 0, z)

def tanh(z):
	return T.tanh(z)

def loadSeqData(TrainFile, LabelFile, mapping):
	#for data part
	X = pickle.load(open(TrainFile, "rb"))
	Xseq = []
	for seq in X:
		seq = np.matrix(seq)
		Xseq.append(seq)

	#for label part
	Dim = len(mapping)
	
	Y = pickle.load(open(LabelFile, "rb"))
	Yseq = []
	for seq in Y:
		Seq = []
		for label in seq:
			labelvector = [0] * Dim
			labelvector[mapping[label]] = 1
			Seq.append(labelvector)
		Seq = np.matrix(Seq)
		Yseq.append(Seq)

	return Xseq, Yseq

def loadSeqTestData(TestFile):
	#for data part
	X = pickle.load(open(TestFile, "rb"))
	Xseq = []
	for seq in X:
		seq = np.matrix(seq)
		Xseq.append(seq)

	return Xseq

if __name__ == "__main__":

	#-----------------------------------------------------------------#
	#-----------------------loading data------------------------------#

	#to be continued...
	TrainFile = "./SoftmaxTrain.data.pkl"
	LabelFile = "./SoftmaxTrain.lab.pkl"
	MapFile = "./48_39.map"
	
	s = time.time()

	mapping, remapping = makeMapping(MapFile)

	Xseq, Yseq = loadSeqData(TrainFile, LabelFile, mapping)

	print >> sys.stderr, "time: "+str(time.time()-s)
	print >> sys.stderr, time.time(), "done loading training data"

	#-----------------------------------------------------------------#
	#-----------------------build model-------------------------------#



	Neuron_Distribution = [48, 100, 80, 48]

	# params contains: [Neuron distribution, initial learning rate,
	# 					activation function, cost function, epochs]
	params = [Neuron_Distribution, 0.02, "sigmoid", "cross entropy", 1]

	rnn = RNN(params)

	"""
	Here you can decide whether to load model, 
	or just use the initialized one.

	Usage:
		rnn.loadModel(file_path)

	"""
	rnn.buildRNN()

	#-----------------------------------------------------------------#
	#-----------------------training start----------------------------#

	rnn.train(Xseq, Yseq)
	"""
	After training, you can 
	save model like:
		rnn.saveModel(file_path)
	"""

	#-----------------------------------------------------------------#
	#-----------------------testing start-----------------------------#

	rnn.genAns(TestFile, AnsFile)

	TestFile = "SoftmaxTest.data.pkl"
	AnsFile = "./SofmaxTest.ans"
	#do load test data
	Test_seq = loadSeqTestData(TestFile)

	print >> sys.stderr, time.time(), "done loading test data"

	ans = []

	for i in xrange(len(Test_seq)):
		yTest = remap(rnn.testFunc(Test_seq[i]))
		ans += [yTest[i] for i in xrange(yTest.shape[0])]

	print >> sys.stderr, time.time(), "done testing"

	with open(AnsFile, "w") as p:
		for i in ans:
			p.write(remapping[i]+'\n')

	print >> sys.stderr, time.time(), "done writting answer"





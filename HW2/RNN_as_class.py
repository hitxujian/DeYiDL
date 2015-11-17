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
	def __init__(self, params, modelPath=None):
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


		if modelPath == None:
			self.Wi = theano.shared( np.random.uniform(-.1, .1, (48, 256)).astype(dtype='float32'))
			self.Bi = theano.shared( np.random.uniform(-.1, .1, (256)).astype(dtype='float32'))
			self.Wh = theano.shared(.1*np.identity(256).astype(dtype='float32'))#to identity matrix
			self.Bh = theano.shared(np.zeros(256).astype(dtype='float32'))
			self.Wo = theano.shared( np.random.uniform(-.1, .1, (256, 48)).astype(dtype='float32'))
			self.Bo = theano.shared( np.random.uniform(-.1, .1, (48)).astype(dtype='float32'))
			self.WB_parameters = [self.Wi, self.Bi, self.Wh, self.Bh, self.Wo, self.Bo]
		
		else:
			self.loadModel(modelPath)

		self.h  = theano.shared(np.zeros(256).astype(dtype='float32'))
		self.y  = theano.shared(np.zeros(48).astype(dtype='float32'))
		self.INIT_parameters = [self.h, self.y]


	def loadModel(self, fpath):
		f = file(fpath, 'rb')
		model = pickle.load(f)
		self.Wi = theano.shared(model[0].get_value())
		self.Bi = theano.shared(model[1].get_value())
		self.Wh = theano.shared(model[2].get_value())
		self.Bh = theano.shared(model[3].get_value())
		self.Wo = theano.shared(model[4].get_value())
		self.Bo = theano.shared(model[5].get_value())
		self.WB_parameters = [self.Wi, self.Bi, self.Wh, self.Bh, self.Wo, self.Bo]
		f.close()

	def saveModel(self, fpath):
		pickle.dump(self.WB_parameters, open(fpath, "wb")) 
		

	def buildRNN(self):
		x_seq = T.matrix('input')
		y_hat = T.matrix('target')
			
		def step(x_t, h_tm1, y_tm1):
			z_t = T.dot(x_t, self.Wi) + self.Bi \
				+ T.dot(h_tm1, self.Wh) + self.Bh
			# h_t = sigmoid(z_t)
			# h_t = T.switch(z_t<0,0.0001,z_t)
			h_t = self.activation(z_t)

			zy_t = T.dot(h_t, self.Wo) + self.Bo
			y_t = softmax(zy_t)
			return h_t, y_t
		
		
		[h_seq, y_seq],_ = theano.scan(
							step, \
							sequences = x_seq, \
							outputs_info =  [self.h, self.y], \
							truncate_gradient=-1 \
		)

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


		def RMSprop(cost, params, lr=0.00005, rho=0.9, epsilon=1e-6):
			grads = T.grad(cost=cost, wrt=params)
			updates = []
			for p, g in zip(params, grads):
				acc = theano.shared(p.get_value()*0.)

				acc_new = rho * acc + (1 - rho) * g ** 2
				gradient_scaling = T.sqrt(acc_new + epsilon)
				g = g / gradient_scaling
				updates.append((acc, acc_new))
				updates.append((p, p - lr*T.clip(g,-5, 5)))
			return updates

		test = theano.function(
			inputs= [x_seq], \
			outputs=y_seq_last \
		)

		train = theano.function(
				inputs=[x_seq,y_hat], \
				outputs=self.costFunc, \
				updates=RMSprop(self.costFunc, self.WB_parameters) \
		)
		'''
		Lambda = np.float32(1.)
		
		#add a momentum
		nag = theano.function(
				inputs=[], \
				on_unused_input="warn", \
				# v = last w
				updates=[(w, w + Lambda * (w - v)) for w, v in izip(self.WB_parameters, self.V_parameters)]
			)
		#update v = last w
		nagv = theano.function(
				inputs=[], \
				on_unused_input = "warn", \
				updates=[(v, (1*w + Lambda*v)/(1+Lambda)) for v, w in izip(self.V_parameters, self.WB_parameters)]
			)

		
		self.nag = nag
		self.nagv = nagv
		'''
		self.testFunc = test
		self.trainFunc = train
		print >> sys.stderr, "Done building RNN"


	def train(self, data, labels):
		lastEin = 1
		for t in range(self.epochs):
			
			cost = 0.
			s = time.time()

			for i in xrange(len(data)):
				#self.nag()
				#self.nagv()
				cost += self.trainFunc(data[i], labels[i])

			if np.isnan(cost):
				print "NAN problem"
				break


			if t % 5 == 0:
				print >> sys.stderr, "----------- iteration:", t,"----------"
				print >> sys.stderr, "cost:", cost
				
				#count Ein...
				error, errorRate = self.test(data, labels)
				print >> sys.stderr, "Ein: " + str(errorRate) + "\n\n"

				if errorRate < lastEin:
					print "Smaller than last Ein, save model."
					lastEin = errorRate
					self.saveModel("myModel.pkl")
			


		print >> sys.stderr, time.time(), "Done training"

			

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
		

	


	def MyUpdate(self, parameters, gradients):
		parameters_updates = [(p, p - self.mu * g) for p,g in izip(parameters, gradients)] 
		return parameters_updates


	def valid(self, ySeq, yHatSeq):

		#here yBatch is a batch, not a list of batches
		error = 0.
		for i in xrange(len(yHatSeq)):
			#because y is array, but yHat is matrix
			y = np.argmax(ySeq[i])
			yHat = np.argmax(yHatSeq[i])
			if y != yHat:
				#0/1 error type
				error += 1.

		return error

	def remap(self, yTest):
		
		y = np.argmax(yTest, axis=1)

		return y



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
	return T.exp(z) / T.sum(T.exp(z))

def ReLU(z):
	return T.switch(z<0, 0.001, z)

def tanh(z):
	return T.tanh(z)

def loadSeqData(TrainFile, LabelFile, mapping):
	#for data part
	X = pickle.load(open(TrainFile, "rb"))
	Xseq = []
	for seq in X:
		seq = np.array(seq,dtype='float32')
		Xseq.append(seq)

	#for label part
	Dim = len(mapping)
	
	Y = pickle.load(open(LabelFile, "rb"))
	Yseq = []
	for seq in Y:
		Seq = []
		for label in seq:
			labelvector = [0.] * Dim
			labelvector[mapping[label]] = 1.
			Seq.append(labelvector)
		Seq = np.array(Seq,dtype='float32')
		Yseq.append(Seq)

	return Xseq, Yseq

def loadSeqTestData(TestFile):
	#for data part
	X = pickle.load(open(TestFile, "rb"))
	Xseq = []
	for seq in X:
		seq = np.array(seq,dtype='float32')
		Xseq.append(seq)

	return Xseq

if __name__ == "__main__":

	#-----------------------------------------------------------------#
	#-----------------------loading data------------------------------#

	#to be continued...
	TrainFile = "./train10.data.pkl"
	LabelFile = "./train10.lab.pkl"
	MapFile = "./48_39.map"
	
	s = time.time()

	mapping, remapping = makeMapping(MapFile)

	Xseq, Yseq = loadSeqData(TrainFile, LabelFile, mapping)

	print >> sys.stderr, "time: "+str(time.time()-s)
	print >> sys.stderr, time.time(), "done loading training data"

	#-----------------------------------------------------------------#
	#-----------------------build model-------------------------------#



	Neuron_Distribution = [48, 256, 48]

	# params contains: [Neuron distribution, initial learning rate,
	# 					activation function, cost function, epochs]
	params = [Neuron_Distribution, 0.1, "ReLU", "cross entropy", 200]

	rnn = RNN(params, "myModel.pkl")
	
	#rnn.loadModel("myModel.pkl")


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
	"""
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

	"""



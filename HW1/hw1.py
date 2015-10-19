import theano
import theano.tensor as T 
import numpy
import random
from itertools import izip
import time

if __name__ == "__main__" :

	def MyUpdate(paramaters, gradients):
		mu = 0.1
		paramaters_update = \
		[(p, p - mu * g) for p, g in izip(paramaters, gradients) ]
		return paramaters_update
	
	x = T.matrix()
	w = theano.shared(numpy.asmatrix(numpy.ones((4,3))))
	b = theano.shared(numpy.ones(4))

	print x.type

	wy = theano.shared(numpy.asmatrix(numpy.ones((3,4))))
	
	a1 = 1/(1+T.exp(-1*(T.dot(w,x) + b.dimshuffle(0,'x'))))
	y = 1/(1+T.exp(-1*(T.dot(wy, a1))))
	
	y_hat = T.matrix()
	cost = T.sum((y-y_hat)**2)

	gradients = T.grad(cost, [w, b, wy])

	neuron1 = theano.function(inputs=[x],outputs=a1)
	neuron2 = theano.function(inputs=[a1],outputs=y)

	train = theano.function(inputs=[x, y_hat], updates=MyUpdate([w, b, wy], gradients), outputs=cost)
	test = theano.function(inputs=[x], outputs=y)
	x = numpy.matrix([[1,1],[-1,1],[1,1]], dtype='float32')#[[1, -1, 1],[1, 1, 1]]
	#x = numpy.array(x).astype(dtype='float32')
	#print x.type.dtype
	y_hat = numpy.matrix([[0,1],[1,0],[0,0]], dtype='float32')#[[0, 1, 0],[1, 0, 0]]
	#x = numpy.array(y_hat).astype(dtype='float32')

	
	s = time.time()
	
	for i in range(1000):
		print "--------"+str(i+1)+"--------"
		print train(x, y_hat)
		print test(x)
		print w.get_value(), b.get_value()
		print wy.get_value()
	f = time.time()

	print f-s

	
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
 
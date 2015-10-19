#### This is only for showing the speedup for GPU

import theano
import theano.tensor as T
import numpy
import time

X = T.matrix(dtype='flaot32')
Y = T.matrix(dtype='float32')
Z = T.dot(X, Y)
f = theano.function([X, Y], Z)

x = numpy.random.randn(10000, 10000).astype(dtype='float32')
y = numpy.random.randn(10000, 10000).astype(dtype='float32')
tStart = time.time()
z = f(x, y)
tEnd = time.time()
print "It cost %f sec" % (tEnd - tStart)
#import theano
#import theano.tensor as T
import numpy
import random

# batchNumber represents how many data are in one batch
# dataSize represents how many features in one data
def mkBatch(xAll, yHatAll, dataSize, batchNumber):
	xBatch = []
	yHatBatch = []
	index = 0
	batchCnt = 0
	allData = len(yHatAll)
	flag = False
	while flag == False:
		if index >= allData:
			break
		xBatch.append([])
		yHatBatch.append([])

		# for yHatBatch
		for i in range(48):
			yHatBatch[batchCnt].append([])
			for j in range(index, index + batchNumber):
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
			for j in range(index, index + batchNumber):
				if j >= allData:
					flag = True
					break
				xBatch[batchCnt][i].append(xAll[j][i])
			if flag:
				break
		index += batchNumber
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






trainFile = open("miniData", "r")
labelFile = open("miniTrain", "r")
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


dataSize = len(xAll[0])
xBatch, yHatBatch = mkBatch(xAll, yHatAll, dataSize, 10)

print xBatch
print yHatBatch
















"""
x = T.vector()
w = theano.shared(numpy.array([-1., 1.]))
b = theano.shared(0.)

z = T.dot(w, x) + b
y = 1 / (1 + T.exp(-z))

neuron = theano.function([x], y)

y_hat = T.scalar()
cost = T.sum((y - y_hat) ** 2)
dw, db = T.grad(cost, [w, b])
gradient = theano.function(inputs=[x, y_hat], updates=[(w, w - 0.1 * dw), (b, b - 0.1 * db)])


x = [1, -1]
y_hat = 1

for i in range(100):
	
	print neuron(x)
	dw, db = gradient(x, y_hat)
	w.set_value(w.get_value() - 0.1 * dw)
	b.set_value(b.get_value() - 0.1 * db)
	print w.get_value(), b.get_value()

"""
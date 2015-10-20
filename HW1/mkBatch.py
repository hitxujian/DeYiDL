#import theano
#import theano.tensor as T
import numpy
import random

# batchNumber represents how many batches there are
# dataSize represents how many features in one data
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


#import theano
#import theano.tensor as T
import numpy
import random

def writeToFile(data, remapping, batchSize, testingIDs, file):
	for i in range(batchSize):
		maxValue = 0
		maxIndex = 0
		for j in range(48):
			if data[i][j] > maxValue:
				maxValue = data[i][j]
				maxIndex = j
		f.write()

# batchNumber represents how many batches there are
# dataSize represents how many features in one data
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






trainFile = open("miniData", "r")
labelFile = open("miniTrain", "r")
mapFile = open("48_39.map", "r")

outputFile = open("TESTFILE", "w+")

mapping, remapping = makeMapping(mapFile)

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

writeToFile()

#print xBatch
#print yHatBatch
#print remapping[1]


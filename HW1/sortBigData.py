import operator

trainFile = open("../../HW1/MLDS_HW1_RELEASE_v1/fbank/train.ark", "r")
labelFile = open("../../HW1/MLDS_HW1_RELEASE_v1/label/train.lab", "r")


newTrainFile = open("newTrain.ark", "w")
newLabelFile = open("newTrain.lab", "w")

allTrainData = []
allLabelData = []

while True:
	tmpLine = trainFile.readline().strip()
	if tmpLine == "":
		break
	allTrainData.append(tmpLine.split())

	tmpLine = labelFile.readline().strip()
	allLabelData.append(tmpLine.split(","))






sortedData = sorted(allTrainData, key=operator.itemgetter(0))


print "trainFile Sorted"


for i in sortedData:
	for j in i:
		newTrainFile.write(str(j) + " ")
	newTrainFile.write("\n")

print "newTrainFile written"

sortedLabel = sorted(allLabelData, key=operator.itemgetter(0))

print "trainLabel Sorted"

for i in sortedLabel:
	for j in i:
		newLabelFile.write(str(j) + " ")
	newLabelFile.write("\n")
	
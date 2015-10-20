import random

trainFile = open("newTrain.ark", "r")
labelFile = open("newTrain.lab", "r")

TrainData = open("TrainData", "w")
ValData = open("ValidationData", "w")

TrainLabel = open("TrainLabel", "w")
ValLabel = open("ValidationLabel", "w")


while True:
	tmpTrainLine = trainFile.readline()
	if tmpTrainLine == "":
		break




	tmpLabelLine = labelFile.readline()


	rand = random.randrange(100)
	if rand < 20:
		ValData.write(tmpTrainLine)
		ValLabel.write(tmpLabelLine)
	else:
		TrainData.write(tmpTrainLine)
		TrainLabel.write(tmpLabelLine)
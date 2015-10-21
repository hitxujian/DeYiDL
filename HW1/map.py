import sys

testingFile = open("MLDS/fbank/test.ark", "r")
resultFile = open("test_ans", "r")

outputFile = open("output2.csv", "w")

outputFile.write("ID,Prediction\n")

while True:
	tmpLine = testingFile.readline()
	if tmpLine == "":
		break
	ID = tmpLine.split()[0]
	outputFile.write(ID + ",")
	tmpLine = resultFile.readline().strip()
	outputFile.write(tmpLine + "\n")
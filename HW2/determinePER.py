import numpy as np


def LevenshteinDistance(groundTruth, myAns):
	d_matrix = np.zeros((len(groundTruth), len(myAns)))
	for i in range(len(groundTruth)):
		d_matrix[i][0] = i

	for j in range(len(myAns)):
		d_matrix[0][j] = j

	for j in range(len(myAns)):
		for i in range(len(groundTruth)):
			if groundTruth[i] == myAns[j]:
				d_matrix[i][j] = d_matrix[i - 1][j - 1]
			else:
				d_matrix[i][j] = min(d_matrix[i - 1][j] + 1, 
									 d_matrix[i][j - 1] + 1, 
									 d_matrix[i - 1][j - 1] + 1)
	return d_matrix[len(groundTruth) - 1][len(myAns) - 1]



"""
groundTruth contains all sentences and its labels
"""
def determinePER(groundTruth, myAns):
	dist = 0.0
	for i in range(len(groundTruth)):
		dist += LevenshteinDistance(groundTruth[i], myAns[i])

	avgDist = dist / len(groundTruth)
	return avgDist

"""
This can be replaced by label sequence
"""

if __name__ == "__main__":
	groundTruth = [5, 7, 3, 4, 1, 1, 6, 3, 2, 0, 8]
	myAns = [5, 5, 7, 1, 3, 4, 1, 9, 3, 2, 6, 8]
	print LevenshteinDistance(groundTruth, myAns)





	
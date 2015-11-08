import numpy as np


def LevenshteinDistance(groundTruth, myAns):
	d_matrix = np.zeros((len(groundTruth) + 1, len(myAns) + 1))
	for i in range(len(groundTruth) + 1):
		d_matrix[i][0] = i

	for j in range(len(myAns) + 1):
		d_matrix[0][j] = j

	for j in range(1, len(myAns) + 1):
		for i in range(1, len(groundTruth) + 1):
			if groundTruth[i - 1] == myAns[j - 1]:
				d_matrix[i][j] = d_matrix[i - 1][j - 1]
			else:
				d_matrix[i][j] = min(d_matrix[i - 1][j] + 1, 
									 d_matrix[i][j - 1] + 1, 
									 d_matrix[i - 1][j - 1] + 1)
	"""
	for i in range(len(groundTruth)):
		for j in range(len(myAns)):
			print d_matrix[i][j], 
		print ""
	"""
	return d_matrix[len(groundTruth)][len(myAns)]



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
	groundTruth = [1,2,3,4,5]
	myAns = [5,4,3,3,5,1,2,3,3,5,2,1]
	print LevenshteinDistance(groundTruth, myAns)






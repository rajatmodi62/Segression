# check if it is possible to go to position (x, y) from
# current position. The function returns false if the cell
# has value 0 or it is already visited.
def isSafe(mat, visited, x, y):
	return not (mat[x][y] == 0 or visited[x][y])


# if not a valid position, return false
def isValid(x, y,M,N):
	return M > x >= 0 and N > y >= 0


# Find Longest Possible Route in a Matrix mat from source
# cell (0, 0) to destination cell (x, y)
# 'max_dist' stores length of longest path from source to
# destination found so far and 'dist' maintains length of path from
# source cell to the current cell (i, j)
def findLongestPath(mat, visited, i, j, x, y,M,N, max_dist, dist,path,longest_path):
	path=path.copy()
	path.append((i,j))
	# if destination not possible from current cell
	if mat[i][j] == 0:
		return longest_path,0

	# if destination is found, update max_dist
	if i == x and j == y:
		#print("len of path",len(path))
		if dist>max_dist:
			#print("true")
			return path,max(dist, max_dist)
		else:
			return longest_path,max_dist
	# set (i, j) cell as visited
	visited[i][j] = 1

	# go to bottom cell
	if isValid(i + 1, j,M,N) and isSafe(mat, visited, i + 1, j):
		longest_path,max_dist = findLongestPath(mat, visited, i + 1, j, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to right cell
	if isValid(i, j + 1,M,N) and isSafe(mat, visited, i, j + 1):
		longest_path,max_dist = findLongestPath(mat, visited, i, j + 1, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to top cell
	if isValid(i - 1, j,M,N) and isSafe(mat, visited, i - 1, j):
		longest_path,max_dist = findLongestPath(mat, visited, i - 1, j, x, y, M,N,max_dist, dist + 1,path,longest_path)

	# go to left cell
	if isValid(i, j - 1,M,N) and isSafe(mat, visited, i, j - 1):
		longest_path,max_dist = findLongestPath(mat, visited, i, j - 1, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to top-left cell
	if isValid(i-1, j - 1,M,N) and isSafe(mat, visited, i-1, j - 1):
		longest_path,max_dist = findLongestPath(mat, visited, i-1, j - 1, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to top-right cell
	if isValid(i-1, j + 1,M,N) and isSafe(mat, visited, i-1, j + 1):
		longest_path,max_dist = findLongestPath(mat, visited, i-1, j + 1, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to bottom-left cell
	if isValid(i+1, j - 1,M,N) and isSafe(mat, visited, i+1, j - 1):
		longest_path,max_dist = findLongestPath(mat, visited, i+1, j - 1, x, y,M,N, max_dist, dist + 1,path,longest_path)

	# go to bottom-right cell
	if isValid(i+1, j + 1,M,N) and isSafe(mat, visited, i+1, j + 1):
		longest_path,max_dist = findLongestPath(mat, visited, i+1, j + 1, x, y,M,N, max_dist, dist + 1,path,longest_path)
	# Backtrack - Remove (i, j) from visited matrix
	visited[i][j] = 0

	return longest_path,max_dist


if __name__ == '__main__':
	import numpy as np
	import sys
	np.set_printoptions(threshold=sys.maxsize)
 
	with open('testing_obsolete/test.npy', 'rb') as f:
		a = np.load(f)
	#print(a[:200,:200])
	import matplotlib.pyplot as plt
	plt.imshow(a)
	plt.show()
	
	# 	# input matrix
	# mat = [
	# 	[1,0,0,0,0,0,0,0,0,0],
	# 	[0,1,0,0,0,0,0,0,0,0],
	# 	[0,1,1,0,0,0,0,0,0,0],
	# 	[0,1,0,0,0,0,0,0,0,0],
	# 	[0,0,1,0,0,0,0,0,0,0],
	# 	[0,0,0,1,0,0,0,0,0,0],
	# 	[0,0,0,0,0,0,0,0,0,0],
	# 	[0,0,0,0,0,0,0,0,0,0],
	# 	[0,0,0,0,0,0,0,0,0,0],
	# 	[0,0,0,0,0,0,0,0,0,0]
		
	# ]
	# # input matrix
	# mat = [
	# 	[1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
	# 	[1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
	# 	[1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
	# 	[0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
	# 	[1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
	# 	[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	# 	[1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
	# 	[1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
	# 	[1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
	# 	[1, 0, 1, 1, 1, 1, 0, 1, 0, 0]
	# ]

	# M x N matrix
	# M = N = 10

	# # construct a matrix to keep track of visited cells
	# visited = [[0 for x in range(N)] for y in range(M)]

	# # (0, 0) are the source cell coordinates and (5, 7) are the
	# # destination cell coordinates
	# longest_path,max_dist = findLongestPath(mat, visited, 0, 0, 5, 3,M,N, 0, 0,[],[])

	# print("Maximum length path is", longest_path,max_dist)
import numpy as np
#M = np.array([[10**-16, 1, 2], [1,1,3]])
M = np.array([[0,1,2],[1,1,3]])

x = np.array([0,0])
n = len(M)
print(n)
for i in range(len(M) - 1):
	for j in range(i+1, len(M)):
		if M[i][i] != 0:
			scalar = M[i][j]/M[i][i]
			# M[j] = [M[j][k] - scalar*M[i][k] for k in range(len(M[0]))]
			M[j] = M[j] - scalar*M[i]
print(M)
print("first pass done")

x[n-1] = M[n-1][n]/M[n-1][n-1]

for i in range(n-2, -1, -1):
	x[i] = M[i][n]
	for j in range(i+1,n):
		x[i] = x[i] - M[i][j]*x[j] 
	x[i] = x[i]/M[i][i]

print(x)

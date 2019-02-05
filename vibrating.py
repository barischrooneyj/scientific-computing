import math
import matplotlib.pyplot as plt

def Phifunction(x):
	return math.sin(2 * math.pi * x)

print(math.sin(0.5 * math.pi))
L = 1
N = 500
endTime = 400
c = 1
deltaT = 0.001

stringMatrix = []
startString = [0] * (N+1)

for i in range(N+1):
	if(i == 0 or i == N):
		pass
	else:
		startString[i] = Phifunction(L/N*i) 
	
print(startString)
	
stringMatrix.append(startString)
stringMatrix.append(startString)

for j in range(endTime):
	tempString = [0] * (N+1)
	for i in range(N + 1):
		if(i == 0 or i == N):
			tempString[i] = 0
		else:
			tempString[i] = ( deltaT * c / (L/N) ) ** 2 * (stringMatrix[-1][i+1] + stringMatrix[-1][i-1] - 2 *stringMatrix[-1][i]) - stringMatrix[-2][i] + 2 * stringMatrix[-1][i]
	stringMatrix.append(tempString)


print(stringMatrix[-1])
	
color = ["red", "orange", "yellow", "green", "blue"]
	
plt.figure()
for i in range(5):
	plt.plot(range(N + 1), stringMatrix[round(endTime / (5 - i))], c=color[i])
plt.show()
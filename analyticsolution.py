from scipy.stats import norm
import matplotlib.pyplot as plt
import math

D = 1

def c(x,t):
	answer = 0
	for i in range(100):
		newit = math.erfc((1-x+2*i)/(2*math.sqrt(D*t))) - math.erfc((1+x+2*i)/(2*math.sqrt(D*t)))
		answer += newit
	return answer
	
N=100
a = [i/N for i in range(N+1)]

print(c(0.125, 0.5))

plt.figure()
for i in range(4):
	list = [c(j,1/(10.0**i)) for j in a]
	plt.plot(a, list, label = 1/(10.0**i))
	
plt.legend()
plt.show()
import numpy as np


L = 1
dx = 0.05
dy = 0.05
Ni = int(L/dx)
Nj = int(L/dy)
if Ni != L/dx: raise Exception("Invalid dx value")


M = np.zeros((Ni * Nj, Ni * Nj))
v = np.zeros((Nj * Ni, 1))


irange = range(Ni)
jrange = range(Nj)

def boundary(i, j, Ni = Ni, Nj = Nj):
    if i == 0 or i == Ni - 1:
        return True
    
    elif j == 0 or j == Nj - 1:
        return True

    else:
        return False
        
for i in irange:
    for j in jrange:
        if not boundary(i, j):

            print(i , j , "Special")
            M[i * Ni + j, i * Ni + j] = -4
            M[i * Ni + j, (i - 1) * Ni + j] = 1
            M[i * Ni + j, (i + 1) * Ni + j] = 1
            M[i * Ni + j, i * Ni + j - 1] = 1
            M[i * Ni + j, i * Ni + j + 1] = 1
            

            

#print(M)           

answer = np.linalg.eig(M * 1/dx**2)

print(answer[0])
print()
print(answer[1])

#for i in range(len(answer[0])):
#    if abs(answer[0][i]) <0.01:
#        print(answer[0][i], answer[1][i], np.dot(M, answer[1][i]))
        
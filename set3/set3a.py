import numpy as np


Ni = 4
Nj = 4

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
        if boundary(i, j):
            M[i, j * i] = 0
            
            print(i , j , "Boundary")
            
        else:
            print(i , j , "Special")
            M[i * Ni + j, i * Ni + j] = -4
            M[i * Ni + j, (i - 1) * Ni + j] = 1
            M[i * Ni + j, (i + 1) * Ni + j] = 1
            M[i * Ni + j, i * Ni + j - 1] = 1
            M[i * Ni + j, i * Ni + j + 1] = 1
            

            

print(M)           

#answer = np.linalg.eig(M)
#
#for i in range(len(answer[0])):
#    if abs(abs(answer[0][i]) - 1) <0.0001:
#        print(answer[0][i], answer[1][i], np.dot(M, answer[1][i]))
#        
import numpy as np
import scipy.linalg as linalg

import matplotlib.pyplot as plt

def makeMatrixM(Ni, alpha):
    M = np.zeros((Ni ** 2, Ni ** 2))

    for i in range(Ni):
        for j in range(Ni):
            if not circleBoundary(i, j, Ni):
                
                    try: M[i * Ni + j, i * Ni + j] = 1-4*alpha 
                    except: pass
                    try: M[i * Ni + j, (i + 1) * Ni + j] = alpha 
                    except: pass
                    if (i - 1) * Ni + j >= 0 : 
                        M[i * Ni + j, (i - 1) * Ni + j] = alpha 
                    if j != Ni-1:
                        M[i * Ni + j, i * Ni + (j + 1)] = alpha 
                    
                    if i * Ni + (j - 1) >= 0: 
                        M[i * Ni + j, i * Ni + (j - 1)] = alpha 
                    

    return M
    
    
def circleBoundary(i, j, Ni):
    if Ni % 2 == 0:
        raise ValueError("Circle does not have odd length")
    C = Ni // 2
    dist = np.sqrt(abs(i - C) ** 2 + abs(j - C) ** 2)
    r = (Ni) / 2
    return dist > r  

if __name__ == "__main__":

    L = 1
    Ni = 5
    deltax = 4./Ni
    D = 1
    
    alpha = 0.05

    M = makeMatrixM(Ni, alpha) 
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    b = np.zeros(Ni**2)
    b[8] = 1
    for i in range(40):
        print(i)
        b = np.dot(M, b)
        b[8] = 1
    
    b.shape = (Ni, Ni)
    plt.imshow(b)
    plt.colorbar()
    plt.show()
    
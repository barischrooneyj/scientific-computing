import numpy as np
import scipy.linalg as linalg

import matplotlib.pyplot as plt

def makeMatrixM(Ni, alpha):
    M = np.zeros((Ni ** 2, Ni ** 2))

    for i in range(Ni):
        for j in range(Ni):
            if not circleBoundary(i, j, Ni):
                
                    try: M[i * Ni + j, i * Ni + j] = -4*alpha 
                    except: pass
                    try: M[i * Ni + j, (i + 1) * Ni + j] = alpha 
                    except: pass
                    if (i - 1) * Ni + j >= 0 : 
                        M[i * Ni + j, (i - 1) * Ni + j] = alpha 
                    if j != Ni-1:
                        M[i * Ni + j, i * Ni + (j + 1)] = alpha 
                    
                    if i * Ni + (j - 1) >= 0: 
                        M[i * Ni + j, i * Ni + (j - 1)] = alpha 
            else:
                M[i * Ni + j, i * Ni + j] = 1
                    

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
    Ni = 21
    deltax = 4./Ni
    D = 1
    
    alpha = 1/(deltax ** 2)

    isource = int(2.6/4*Ni)
    jsource = int(0.8/4*Ni)
    vec_source_space = jsource * Ni + isource
    M = makeMatrixM(Ni, alpha) 
    plt.imshow(M)
    plt.colorbar()
    plt.show()
    b = np.zeros(Ni**2)
    b[vec_source_space] = 1
    M[vec_source_space] = np.zeros((1, len(M[vec_source_space])))
    
    M[vec_source_space][vec_source_space] = 1
    
    plt.imshow(M)
    plt.colorbar()
    plt.show()    

    c = linalg.solve(M, b)
    plt.imshow(c.reshape((Ni, Ni)))
    plt.colorbar()
    plt.show()
    
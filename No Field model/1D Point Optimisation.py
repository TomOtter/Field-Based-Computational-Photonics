import numpy as np 
import scipy
from matplotlib import pyplot as plt

#Define size of input vector and scattering matrix
n = 256

pi = np.pi

# SLM

real_part = np.random.uniform(0, 1, int(n/2))
real_part_data = np.sin(np.arange(int(n/2), n) * 0.05)**2

# plt.plot(np.arange(0,int(n/2),1),real_part_data)
# plt.show()

imaginary_part = np.random.uniform(-0, 0, n)
slm_matrix = np.concatenate((real_part,real_part_data)) + 1j * imaginary_part


# Scatterer
# random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
# scattering_matrix, _ = np.linalg.qr(random_matrix)  # QR decomposition to make it unitary

scattering_matrix = scipy.stats.unitary_group.rvs(n)

scattered = scattering_matrix @ slm_matrix

# Measure loop
sinx = np.sin(np.arange(0, n//2) * 0.05)/2 + 0.5
imaginary_part = np.random.uniform(-0, 0, n)


#Test scatter:
# test_slm = np.ones(n)
# output = abs(scattering_matrix @ test_slm)**2
# plt.plot(output)
# plt.xlabel(r'Position (Arbitary Distance)')
# plt.ylabel(r'Intensity output (Arbitary $W/m^2$)')
# plt.show()

def scatter_sinx(adjust_part):
    slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    scattered = scattering_matrix @ slm_matrix
    return -abs(scattered[int(n/2)])**2

bounds = [(0, 1)] * (n//2)
result = scipy.optimize.minimize(scatter_sinx, real_part, bounds=bounds, method='L-BFGS-B')



def scatter_sinx_fullresult(adjust_part):  
    slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    plt.plot(sinx)
    plt.show()
    scattered = scattering_matrix @ slm_matrix
    return abs(scattered)**2


full_output = scatter_sinx_fullresult(result.x)
plt.plot(np.arange(0,int(n),1),full_output)
plt.xlabel(r'Position (Arbitary Distance)')
plt.ylabel(r'Intensity Output (Arbitary $W/m^2$)')
plt.show()
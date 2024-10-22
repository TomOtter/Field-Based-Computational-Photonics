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
sinx = np.sin(np.arange(int(n/2), n) * 0.05)/2 + 0.5
imaginary_part = np.random.uniform(-0, 0, n)

def scatter_sinx(adjust_part):
    slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    scattered = scattering_matrix @ slm_matrix
    return -abs(scattered[int(n/2)])**2

bounds = [(0, 1)] * int(n/2)
result = scipy.optimize.minimize(scatter_sinx, real_part, bounds=bounds, method='L-BFGS-B')



def scatter_sinx_fullresult(adjust_part):  
    slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    scattered = scattering_matrix @ slm_matrix
    return abs(scattered)**2


# full_output = scatter_sinx_fullresult(result.x)
# plt.plot(np.arange(0,int(n),1),full_output)
# plt.show()


def new_error_function(adjust_part):
    slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    scattered = scattering_matrix @ slm_matrix

    desired_func = np.sin(np.arange(0,n,1)*0.1)
    error = np.mean((abs(scattered) - desired_func)**2)
    return error

# result = scipy.optimize.minimize(new_error_function, real_part, bounds=bounds, method='L-BFGS-B', options={"maxfun":1e9})
result = scipy.optimize.differential_evolution(new_error_function, bounds=bounds, disp="True")
full_output = abs(scattering_matrix @ np.concatenate((result.x,sinx)))**2
plt.plot(np.arange(0,int(n),1),full_output)
plt.show()

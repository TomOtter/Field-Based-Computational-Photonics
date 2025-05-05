import numpy as np 
import scipy
from matplotlib import pyplot as plt

#Define size of input vector and scattering matrix
n = 512

pi = np.pi

# SLM

#real_part = np.random.uniform(0, 1, int(n/2))
real_part = np.random.uniform(0, 1, int(n))

#real_part_data = np.sin(np.arange(int(n/2), n) * 0.05)**2

# plt.plot(np.arange(0,int(n/2),1),real_part_data)
# plt.show()

imaginary_part = np.random.uniform(-0, 0, n)
#slm_matrix = np.concatenate((real_part,real_part_data)) + 1j * imaginary_part
slm_matrix = real_part + 1j * imaginary_part

# Scatterer
# random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
# scattering_matrix, _ = np.linalg.qr(random_matrix)  # QR decomposition to make it unitary

scattering_matrix = scipy.stats.unitary_group.rvs(n)

scattered = scattering_matrix @ slm_matrix
sinx = np.sin(np.arange(0, n//2) * 0.05)/2 + 0.5
imaginary_part = np.random.uniform(-0, 0, n)

#bounds = [(0, 1)] * (n//2)
bounds = [(0, 1)] * (n)


def new_error_function(adjust_part):
    #slm_matrix = np.concatenate((adjust_part,sinx)) + 1j * imaginary_part
    slm_matrix = adjust_part + 1j * imaginary_part
    scattered = scattering_matrix @ slm_matrix

    #desired_func = np.concatenate((np.ones(n//8) , np.zeros(n//8)))
    desired_func = np.sin(np.arange(0,n//8,1) * 0.2) + 1.2
    output = abs(scattered[:n//8])**2
    error = np.sum((output - desired_func)**2)
    return error

result = scipy.optimize.minimize(new_error_function, real_part, bounds=bounds, method='L-BFGS-B', options={"maxfun":1e4})
# result = scipy.optimize.differential_evolution(new_error_function, bounds=bounds, disp="True", maxiter = 20)
# full_output = abs(scattering_matrix @ np.concatenate((result.x,sinx)))**2
full_output = abs(scattering_matrix @ result.x)**2
# plt.scatter(np.arange(0,int(n//8),1),np.sin(np.arange(0,n//8,1) * 0.2) + 1.2, color="orange")
plt.scatter(np.arange(0,int(n//8),1),full_output[:n//8])
plt.xlabel("Spatial position")
plt.ylabel("Intensity")
plt.show()

plt.scatter(np.arange(0,int(n),1),full_output)
plt.xlabel("Spatial position")
plt.ylabel("Intensity")
plt.show()

no_slm_output = abs(scattering_matrix @ np.ones(n))**2
plt.scatter(np.arange(0,int(n),1),no_slm_output)
plt.xlabel("Spatial position")
plt.ylabel("Intensity")
plt.show()
import numpy as np
import scipy.fftpack as fft
from scipy.optimize import minimize
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import scipy
from multiprocessing.dummy import Pool

n_time = 200
n_space = 100
target_position = n_space//2 
pi = np.pi

def random_unitary_tensor(n,d):

    tensor_slices = []
    
    for _ in range(d):
        print("{0}% complete".format(100 * _/d))
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix)
        tensor_slices.append(scatter_matrix)
    print("Done - Initialisation")
    return tensor_slices

scatter_matrix = random_unitary_tensor(n_space,n_time)


# Time waveform input
# ----- Scattering Simulation Setup -----
t = np.linspace(-3, 3, n_time)
input_freq = 5
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)

#Terahertz values
# input_freq = 5e12
# pulse_width = 100e-15
# t = np.linspace(-30e-12, 30e-12, n_time)
# time_domain_waveform = np.exp(1j*input_freq*t) * np.exp(-5*input_freq**2*t**2)

# Objective function based on CF-A
def objective_cf_a(slm_half):
    slm_modulation = slm_half + 1j * np.ones(n_space)
    # slm_modulation[n_space // 2:] = 0.5       # Predetermined function (e.g., constant 0.5)

    #phase modulation
    # for i in range(n_space//2):
    #     slm_modulation[i] *=  np.exp(1j * 2* pi* slm_half[n_space//2 + i])
    
    modulated_input = np.outer(slm_modulation, time_domain_waveform)


    freq_input = fft.fft(modulated_input)

    scatter = []
    for i in range(n_time):
        scatter.append(scatter_matrix[i] @ freq_input[:,i])

    scattered_output = np.stack(scatter, axis=-1)
    output_after_scatterer = fft.ifft(scattered_output)

    output_waveform_at_target = output_after_scatterer[target_position].real


    # Calculate the peak intensity and the temporal standard deviation
    peak_intensity = np.max(output_waveform_at_target)

    #temporal_std_dev = np.std(output_waveform_at_target)

    t = np.linspace(1,200,200)
    e0 = n_time//2
    mu = np.sum(t * abs(output_waveform_at_target)/ e0)


    temporal_std_dev = np.sqrt(np.sum((t-mu)**2 * abs(output_waveform_at_target)/e0))

    

    # Define CF-A as peak intensity divided by temporal standard deviation
    cf_a_value = -peak_intensity / temporal_std_dev



    # Since we want to maximize this, return its negative for minimization
    return cf_a_value

def objective_cf_b(slm_half):
    slm_modulation = np.ones(n_space) + 1j * np.ones(n_space)
    slm_modulation[:n_space // 2] = slm_half[:n_space//2]  # Optimized half
    slm_modulation[n_space // 2:] = 0.5       # Predetermined function (e.g., constant 0.5)

    # phase modulation
    # for i in range(n_space//2):
    #     slm_modulation[i] *=  np.exp(1j * 2* pi* slm_half[n_space//2 + i])
    
    modulated_input = np.outer(slm_modulation, time_domain_waveform)


    freq_input = fft.fft(modulated_input)

    scatter = []
    for i in range(n_time):
        scatter.append(scatter_matrix[i] @ freq_input[:,i])

    scattered_output = np.stack(scatter, axis=-1)
    output_after_scatterer = fft.ifft(scattered_output)

    output_waveform_at_target = output_after_scatterer[target_position].real

    return -np.max(output_waveform_at_target)

# Initial guess for the optimization of the first half of the SLM modulation
initial_slm_half = np.random.uniform(0, 1, n_space)

# Bounds to constrain SLM modulation between 0 and 1
bounds = [(0, 1) for _ in range(n_space)]

# Optimize using CF-A
# result = minimize(objective_cf_a, initial_slm_half, bounds=bounds, method='L-BFGS-B')


with Pool(40) as pool:  # Thread-based Pool
    result =  scipy.optimize.differential_evolution(
        objective_cf_a,
        bounds=bounds, 
        disp="True", 
        strategy='rand1bin',
        popsize=10,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.01,
        maxiter= 5,
        workers=pool.map
    ) 

# Get the optimized SLM configuration
slm_modulation = result.x
# slm_modulation[:n_space] = slm_half #[0:n_space//2 - 1] # + np.exp(2j* pi * slm_half[n_space//2:n_space-1])
#slm_modulation[n_space // 2:] = 0.5 # Predetermined half

# phase mod
# for i in range(n_space//2):
#     slm_modulation[i] *=  np.exp(1j * 2* pi* slm_half[n_space//2 + i])

# Simulate the output with the optimized SLM configuration

modulated_input = np.outer(slm_modulation, time_domain_waveform)


freq_input = fft.fft(modulated_input)

scatter = []
for i in range(n_time):
    scatter.append(scatter_matrix[i] @ freq_input[:,i])

scattered_output = np.stack(scatter, axis=-1)

output_after_scatterer = fft.ifft(scattered_output)

output_waveform_at_target = output_after_scatterer[target_position]


slm_modulation = np.ones(n_space)  + np.ones(n_space) * 0j

modulated_input = np.outer(slm_modulation, time_domain_waveform)


freq_input = fft.fft(modulated_input)

scatter = []
for i in range(n_time):
    scatter.append(scatter_matrix[i] @ freq_input[:,i])

scattered_output = np.stack(scatter, axis=-1)
unmod_output_after_scatterer = fft.ifft(scattered_output)
unmodulated_output_waveform_at_target = unmod_output_after_scatterer[target_position]


plt.subplot(2,1,1)
plt.imshow(modulated_input.real)
plt.subplot(2,1,2)
plt.imshow(unmod_output_after_scatterer.real)
plt.show()




plt.subplot(2,1,1)
plt.plot(output_waveform_at_target)
plt.xlabel("Time, t")
plt.ylabel(r'$E_{out}(t)$')
plt.title("Optimisated modulation for high peak output")
plt.subplot(2,1,2)
plt.plot(unmodulated_output_waveform_at_target)
plt.xlabel("Time, t")
plt.ylabel(r'$E_{out}(t)$')
plt.title("Unmodulated output")
plt.tight_layout()
plt.show()

#Compute averages to see if it is zero
print(np.sum(output_waveform_at_target.real)/n_time)
print(np.sum(unmodulated_output_waveform_at_target.real)/n_time)

# plt.plot(output_waveform_at_target, label="Optimised Output at Target Position")
# plt.plot(unmodulated_output_waveform_at_target, label="Unmodulated Output at Target Position")
# plt.xlabel("Time")
# plt.ylabel("Intensity")
# plt.title("Temporal Waveform at Target Spatial Position")
# plt.legend()
# plt.grid(True)
# plt.show()
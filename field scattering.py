import scipy.fftpack as fft
from matplotlib import pyplot as plt
import numpy as np

def guassian(length, total_indexes):
    center_index=total_indexes//2
    x = np.linspace(0, length - 1, length)
    mean = x[center_index * length // total_indexes]
    std_dev = length / 5
    gaussian_array = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    gaussian_array /= np.max(gaussian_array) # normalisation condition
    return gaussian_array

# Generate scattering matrix (complex weights)
def random_unitary_tensor(n, d):
    #n = no spatial inputs
    #d = no of time
    tensor_slices = []
    p = 0
    add_dispersion = guassian(n,d)
    while p < d:
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix)
        for j in range(n):
            p += 1
            if p > d: break
            tensor_slices.append(scatter_matrix[j] * add_dispersion)
    tensor = np.array(tensor_slices).reshape((d, n))
    return tensor





# --- Inputs ---
n_time = 300
n_space = 200


#1 Define input waveform
input_freq = 5
t = np.linspace(-3, 3, n_time)
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)

plt.plot(time_domain_waveform)
plt.title("Field Input Waveform " + r'$E_{in} (t)$')
plt.xlabel("Time, t")
plt.ylabel(r'$E_{in} (t)$')
plt.tight_layout()

plt.savefig('Field Scattering plots/Field Input Waveform.png')
#
plt.close('all') 

#Create the modulated input so the waveform enters at each spatial position
image_slm = np.ones(n_space)
modulated_input = np.outer(image_slm, time_domain_waveform)


freqs = np.fft.fftfreq(len(time_domain_waveform), d=1/n_time) 
freq_sigma = 20  # Width of Gaussian in frequency domain
f_0 = 25
gaussian_envelope = np.exp(-(freqs-f_0)**2 / (2 * freq_sigma**2))
# Add noise to the Gaussian envelope
noise = np.random.normal(0, 0.01, len(freqs))
noisy_gaussian_envelope = gaussian_envelope + noise
plt.plot(noisy_gaussian_envelope)
plt.title("Noisy guassian envelope " + r'$F(\omega)$')
plt.xlabel("Frequency, " + r'$\omega$')
plt.ylabel(r'$F(\omega)$')
plt.tight_layout()

plt.savefig('Field Scattering plots/Noisy guassian envelope.png')
#
plt.close('all') 

#Multiple the input by the noisy gaussian


# Random unitary complex weights
scatter = random_unitary_tensor(n_space, n_time)



E_fft = fft.fft(modulated_input)
plt.title("Frequency Input Waveform " + r'$\tilde{E_{in}} (\omega)$')
plt.xlabel("Frequencies, w")
plt.ylabel(r'$\tilde{E_{in}} (\omega)$')
plt.plot(E_fft[0])
plt.tight_layout()

plt.savefig('Field Scattering plots/freq input waveform.png')
#
plt.close('all') 


E_fft = E_fft * noisy_gaussian_envelope
plt.title("Frequency Input Waveform parsed by Gaussian Envelope " + r'$\tilde{E_{in}} (\omega)$')
plt.xlabel("Frequencies, w")
plt.ylabel(r'$\tilde{E_{in}} (\omega)$')
plt.plot(E_fft[0])
plt.tight_layout()

plt.savefig('Field Scattering plots/Freq input parsed by envelope.png')
#
plt.close('all') 



# Step 2: column-wise dot product

s = np.einsum('ik,ki->i', scatter, E_fft)

plt.plot(s)
plt.title("Frequency scattered output " + r'$\tilde{E_{out}} (\omega)$')
plt.xlabel("Frequency, " + r'$\omega$')
plt.ylabel(r'$\tilde{E_{out}} (\omega)$')
plt.tight_layout()

plt.savefig('Field Scattering plots/freq scattered output.png')
#
plt.close('all') 

# Step 3: IFFT
s_ifft = fft.ifft(s)
# Step 4: real part, chunk, abs, sum
real = s_ifft.real

plt.plot(real)
plt.title("Output Waveform - Real part of " + r'$E_{out} (t)$')
plt.xlabel("Time, t")
plt.ylabel(r'$E_{out} (t)$')
plt.tight_layout()

plt.savefig('Field Scattering plots/real part of output.png')
#
plt.close('all') 



plt.subplot(2,1,1)
plt.plot(real)
plt.title("Output Waveform - Real part of " + r'$E_{out} (t)$')
plt.xlabel("Time, t")
plt.ylabel(r'$E_{out} (t)$')



x = np.linspace(0,1,n_time)
y = np.sqrt(real **2 + 1e-20)

plt.subplot(2,1,2)
plt.plot(y)
plt.title("Absolute value of real field output " + r'$|Re(E_{out})(t)|$')
plt.xlabel("Time, t")
plt.ylabel(r'$| Re(E_{out})(t)|$')
plt.tight_layout()

plt.savefig('Field Scattering plots/Abs val of field output and real part.png')
#
plt.close('all') 

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y, label='')

# Shade the area between the curve and x-axis
ax.fill_between(x, y, 0, color='lightblue', alpha=0.5)

# Add vertical lines to split into 5 chunks
n_chunks = 10
x_min, x_max = x.min(), x.max()
chunk_width = (x_max - x_min) / n_chunks

for i in range(1, n_chunks):
    ax.axvline(x_min + i * chunk_width, color='gray', linestyle='--')

# Labels and title
ax.set_xlabel('Time, t')
ax.set_ylabel('|Re(E)|')
ax.set_title('Output split into bins')
ax.legend()

plt.tight_layout()
plt.savefig('Field Scattering plots/Output split into bins.png')
plt.close('all') 


bins = np.array_split(y, n_chunks) # split into bins
sum_bins = []
for bin in bins: 
    sum_bins.append(np.sum(bin))
output = np.array(sum_bins)


plt.bar(np.linspace(0,9,10),output)
plt.xlabel("Bin number")
plt.ylabel("Sum of bin")
plt.title("Sum of the output bins")
plt.tight_layout()

plt.savefig('Field Scattering plots/Sum of output bins.png')
plt.close('all') 



plt.imshow(scatter.real)
plt.title("Visulisation of scattering matrix real part")
plt.xlabel("Spatial Position, x")
plt.ylabel("Frequency, w")
plt.tight_layout()
#

plt.savefig('Field Scattering plots/Visualisation of scattering matrix real part.png')
plt.close('all') 

E_fft = fft.fft(modulated_input)
plt.subplot(1,2,1)
plt.imshow(E_fft.real)
E_fft = E_fft * noisy_gaussian_envelope

plt.subplot(1,2,2)
plt.imshow(E_fft.real)
plt.tight_layout()

plt.savefig('Field Scattering plots/input waveform with envelope vs without.png')
plt.close('all') 





# ----- Verify that the distribution of bins for random SLM arrangements is even -----------
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-5 * input_freq**2 * t**2)

sum_outputs = np.zeros(10)
sum_y = np.zeros(n_time)

for i in range(n_space * 100):
    slm = np.random.rand(n_space)
    modulated_input = np.outer(slm, time_domain_waveform)
    E_fft = fft.fft(modulated_input)
    E_fft = E_fft * noisy_gaussian_envelope
    s = np.einsum('ik,ki->i', scatter, E_fft)
    s_ifft = fft.ifft(s)
    real = s_ifft.real
    y = np.sqrt(real **2 + 1e-20)
    bins = np.array_split(y, n_chunks) # split into bins
    sum_bins = []
    for bin in bins: 
        sum_bins.append(np.sum(bin))
    output = np.array(sum_bins)

    sum_y += y
    sum_outputs += output   

plt.subplot(2,1,1)
plt.bar(np.linspace(1,n_time,n_time), sum_y)
plt.xlabel("Time, t")
plt.ylabel("Sum of |Re(E)|")
plt.title("Sum of 200,000 outputs with different random SLMs")


plt.subplot(2,1,2)
plt.bar(np.linspace(1,10,10), sum_outputs)
plt.xlabel("Time, t")
plt.ylabel("Sum of bins")
plt.title("Sum of bin outputs with different random SLMs")
plt.tight_layout()

plt.savefig('Field Scattering plots/sum of 200000 outputs.png')
plt.close('all') 


waterfall = []
slm = np.ones(n_space)
for i in range(n_space):
    scatter = random_unitary_tensor(n_space, n_time)
    modulated_input = np.outer(slm, time_domain_waveform)
    E_fft = fft.fft(modulated_input)
    E_fft = E_fft * noisy_gaussian_envelope
    s = np.einsum('ik,ki->i', scatter, E_fft)
    s_ifft = fft.ifft(s)
    real = s_ifft.real
    waterfall.append(real)


plt.subplot(2,1,1)
plt.imshow(modulated_input.real)
plt.xlabel("Time, t")
plt.ylabel("Spatial position")
plt.title("Input waveform heatmap for sharp pulse")
plt.subplot(2,1,2)
plt.imshow(np.array(waterfall).reshape((n_space,n_time)))
plt.xlabel("Time, t")
plt.ylabel("Spatial position")
plt.title("Scattered output heatmap for sharp pulse")
plt.tight_layout()
plt.savefig('Field Scattering plots/sharp pulse scattered heatmap.png')
#
plt.close('all') 




time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-0.05 * input_freq**2 * t**2)

waterfall = []

slm = np.ones(n_space)
for i in range(n_space):
    scatter = random_unitary_tensor(n_space, n_time)
    modulated_input = np.outer(slm, time_domain_waveform)
    E_fft = fft.fft(modulated_input)
    E_fft = E_fft * noisy_gaussian_envelope
    s = np.einsum('ik,ki->i', scatter, E_fft)
    s_ifft = fft.ifft(s)
    real = s_ifft.real
    waterfall.append(real)

plt.subplot(2,1,1)
plt.imshow(modulated_input.real)
plt.xlabel("Time, t")
plt.ylabel("Spatial position")
plt.title("Input waveform heatmap for broad pulse")
plt.subplot(2,1,2)
plt.imshow(np.array(waterfall).reshape((n_space,n_time)))
plt.xlabel("Time, t")
plt.ylabel("Spatial position")
plt.title("Scattered output heatmap for broad pulse")
plt.tight_layout()

plt.savefig('Field Scattering plots/board pulse scattered heatmap.png')
plt.close('all') 



#test a SLM of zeros gives no output

time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)
slm = np.zeros(n_space)
modulated_input = np.outer(slm, time_domain_waveform)
E_fft = fft.fft(modulated_input)
E_fft = E_fft * noisy_gaussian_envelope
s = np.einsum('ik,ki->i', scatter, E_fft)
s_ifft = fft.ifft(s)
real = s_ifft.real
plt.plot(real)
plt.xlabel("Time, t")
plt.ylabel("Re(E)")
plt.title("E field output for SLM of all zeros - No input")
plt.tight_layout()

plt.savefig('Field Scattering plots/SLM all zeros.png')
plt.close('all') 



from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import csv

my_data = genfromtxt('/home/hicham/Bureau/extraitTest', delimiter=' ')

input_signal = np.reshape(my_data[:,3:].T, -1, 2)
print(input_signal)

# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
FiltreMin = 1
FiltreMax = 10
fs = 333.333
Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]

# b, a = signal.butter(1, 10, 'low', analog=True)
b, a = signal.butter(4,  Wn, 'bandpass')
output_signal = signal.filtfilt(b, a, input_signal)

plt.title("Filtre de Butterworth [1, 10] Hz d'ordre 4")

plt.plot(input_signal, label='Signal')
plt.plot(output_signal, label='Filtre')

plt.legend(bbox_to_anchor=(0.88, 0.88), loc=3, borderaxespad=0.)

plt.show()
"""
#w, h = signal.freqs(b, a)


plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
"""

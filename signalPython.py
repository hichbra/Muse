# -*-coding:Latin-1 -* j

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import csv
import copy

# my_data = genfromtxt('/home/hicham/Bureau/extraitTest', delimiter=' ')
my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv', delimiter=';')


#input_signal = np.reshape(my_data[:,3:].T, -1, 2)
# input_signal = np.reshape(my_data[:,0:1].T, -1, 2) # 1ere colonne
input_signal = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
capteur = np.reshape(my_data[:,27:28].T, -1, 2)
capteur[capteur > 30] = 100
capteur[capteur <= 30] = 0

print(capteur)

# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
# [0.5 30]
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
plt.plot(capteur, label='Capteur')

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
